import os
import json
import torch
import pandas as pd
import numpy as np
import time
import logging
import argparse
import re
import uuid
import csv
import subprocess
import requests
from tqdm import tqdm
from io import BytesIO
from pdf2image import convert_from_path
import base64
from PIL import Image
import traceback
from difflib import SequenceMatcher
import PyPDF2
import pytesseract
import yaml
from openai import OpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from RetrievalAgent import OCRTextAgent, ColQwenVisualAgent
from DecisionAgent import HierarchicalDecisionAgent
from RouterAgent import RouterAgent

try:
    import chromadb
    import chromadb.utils.embedding_functions as embedding_functions
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("apdcrag.log"), logging.StreamHandler()] 
)
logger = logging.getLogger("APDC-RAG")

qa_prompts = {
    "feta_tab": "You are a Wikipedia editor. Answer the question with a single, well-formed, factual sentence.",
    "paper_tab": "You are a research scientist. Answer the question with a concise technical phrase.",
    "scigraphqa": "You are a scientific researcher. Answer the question in 1-2 clear, evidence-based sentences.",
    "slidevqa": "You are a presentation expert. Provide the exact answer to the question as it would appear on a slide. Be direct and precise.",
    "spiqa": "You are a scientific paper author. Answer the question in 1-3 authoritative sentences."
}

class APDCRAG:
    def __init__(self, config):
        """
        Initialize the APDC-RAG pipeline.
        Dual-Engine: Local InternVL3 & Qwen2-VL only.
        Retrievers: ColQwen (Vision) & BGE (Text) only.
        """
        self.config = config
        self.data_dir = config["data_dir"]
        self.output_dir = config["output_dir"]
        self.llm_model = config.get("llm_model", "internvl3").lower()
        self.model_path = config.get("model_path", "/root/autodl-tmp/InternVL3-14B")
        self.vision_retriever = "colqwen"
        self.text_retriever = "bge"
        self.use_router = config.get("use_router", True)
        self.use_gmm = config.get("use_gmm", True)
        self.use_decision = config.get("use_decision", True)
        self.overwrite_results = config.get("overwrite_results", False)
        self.retrieval_pool_size = 20 
        self.top_k_text = 5 
        self.top_k_visual = 5
        self.top_k = config.get("top_k", 5)
        self.vision_retrieval_file = f"{self.data_dir}/retrieval/retrieval_{self.vision_retriever}.csv" 
        self.text_retrieval_file = f"{self.data_dir}/retrieval/retrieval_{self.text_retriever}.csv" 
        self.chunk_size = config.get("chunk_size", 3000)
        self.chunk_overlap = config.get("chunk_overlap", 300)
        self.force_reindex = config.get("force_reindex", False)
        self.dataset_name = os.path.basename(self.data_dir.strip('/'))
        print(f"当前数据集：{self.dataset_name}")
        prompt_config_path = config.get("prompt_config", "./config/prompt.yaml")
        try:
            with open(prompt_config_path, 'r', encoding='utf-8') as f:
                all_prompts = yaml.safe_load(f)
            self.current_prompts = all_prompts.get(self.dataset_name, all_prompts.get("default"))
        except:
            self.current_prompts = {"visual": "{query}", "text": "{query} {context}", "fusion": "Merge: {query}"}

        if self.llm_model == "internvl3":
            self._start_internvl_server()
            self.client = OpenAI(api_key="EMPTY", base_url="http://localhost:23333/v1")
        elif self.llm_model == "qwen":
            self._initialize_qwen()
        else:
            raise ValueError(f"Unsupported LLM model: {self.llm_model}. Must be 'internvl3' or 'qwen'.")

        router_path = config.get("router_model_path", "/root/autodl-tmp/router/t5_router_final_model")
        self.router = RouterAgent(model_path=router_path, use_router=self.use_router)
        self._initialize_retrieval_resources()
        self.text_agent = OCRTextAgent(retrieval_file=self.text_retrieval_file, data_dir=self.data_dir, pool_size=self.retrieval_pool_size, use_gmm=self.use_gmm, build_func=self.build_text_index)
        self.visual_agent = ColQwenVisualAgent(data_dir=self.data_dir, retrieval_file=self.vision_retrieval_file, pool_size=self.retrieval_pool_size, use_gmm=self.use_gmm, build_func=self.build_visual_index)
        self.decision_agent = HierarchicalDecisionAgent(llm_gateway=self, use_decision=self.use_decision)
        self.dataset_csv = config.get("csv_path")
        if not self.dataset_csv:
            self.dataset_csv = f"{self.data_dir}/{os.path.basename(self.data_dir)}.csv"
        for suffix in ["vision", "text", "apdcrag"]:
            os.makedirs(f"{self.output_dir}/{self.llm_model}_{suffix}", exist_ok=True)
        os.makedirs(f"{self.data_dir}/retrieval", exist_ok=True)
        self.df = pd.read_csv(self.dataset_csv)
        self.document_cache = {}
        self.track_visual_tokens = []
        self.track_fusion_tokens = []

    def _initialize_qwen(self):
        logger.info("🚀 Starting Local Qwen2-VL Engine...")
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
            )
            self.qwen_processor = AutoProcessor.from_pretrained(self.model_path)
            self.process_vision_info = process_vision_info
            logger.info("✅ Qwen2-VL Model is ready.")
        except ImportError:
            raise ImportError("Required packages for Qwen not found.")

    def _start_internvl_server(self):
        try:
            if requests.get("http://localhost:23333/v1/models", timeout=2).status_code == 200:
                logger.info("✅ InternVL Server 已经在运行。")
                return
        except:
            pass
        logger.info("🚀 Starting Local InternVL3 Server via LMDeploy...")
        cmd = [
            "lmdeploy", "serve", "api_server", self.model_path, "--model-name", "internvl3",
            "--chat-template", "internvl2_5", "--server-port", "23333", "--tp", "1", 
            "--cache-max-entry-count", "0.8", "--session-len", "32768"
        ]
        self.vlm_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        for _ in range(30):
            try:
                if requests.get("http://localhost:23333/v1/models").status_code == 200:
                    logger.info("✅ InternVL3 Model is ready.")
                    return
            except:
                time.sleep(10)
        logger.error("❌ Failed to start InternVL3 server in time.")

    def __del__(self):
        if hasattr(self, 'vlm_process'):
            self.vlm_process.terminate()

    def _initialize_retrieval_resources(self):
        self.vision_retrieval_file = f"{self.data_dir}/retrieval/retrieval_colqwen.csv"
        self.text_retrieval_file = f"{self.data_dir}/retrieval/retrieval_bge.csv"
        try:
            from colpali_engine.models import ColQwen2, ColQwen2Processor
            logger.info("Loading ColQwen model for visual indexing")
            self.vision_model = ColQwen2.from_pretrained("vidore/colqwen2-v1.0", torch_dtype=torch.bfloat16, device_map="cuda").eval()
            self.vision_processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")
        except ImportError:
            raise ImportError("ColQwen models not found. Please install colpali_engine.")
        logger.info("Loading BGE model for text indexing")
        self.text_model_name = "BAAI/bge-base-en-v1.5"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.st_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.text_model_name, device=self.device)

    def extract_text_from_pdf(self, pdf_path):
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file, strict=False)
                pages = [page.extract_text() for page in reader.pages]
            if any(not page.strip() for page in pages):
                logger.info(f"Using OCR for {pdf_path} as some pages have no text")
                pages = []
                pdf_images = convert_from_path(pdf_path)
                for page_num, page_img in enumerate(pdf_images):
                    text = pytesseract.image_to_string(page_img)
                    pages.append(f"--- Page {page_num + 1} ---\n{text}\n")
            return pages
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return []

    def split_text(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return text_splitter.split_text(text)

    def cache_documents(self):
        logger.info("Caching document content with normalization")
        try:
            unique_docs = set()
            for _, row in self.df.iterrows():
                try:
                    if 'documents' in row:
                        docs = eval(row['documents']) if isinstance(row['documents'], str) else row['documents']
                        unique_docs.update([str(d).replace('.pdf', '') for d in docs])
                except:
                    pass
                if 'doc_path' in row and pd.notna(row['doc_path']):
                    unique_docs.add(os.path.basename(row['doc_path']).replace('.pdf', ''))
            cache = {}
            pdf_dir = os.path.join(self.data_dir, "docs")
            for doc_id in tqdm(unique_docs, desc="Caching documents"):
                for pdf_path in [os.path.join(pdf_dir, f"{doc_id}.pdf"), os.path.join(pdf_dir, doc_id)]:
                    if os.path.exists(pdf_path):
                        cache[doc_id] = self.extract_text_from_pdf(pdf_path)
                        break
            self.document_cache = cache
            return cache
        except Exception as e:
            logger.error(f"Error caching documents: {str(e)}")
            return {}

    def identify_document_and_page(self, chunk):
        max_ratio = 0
        best_match = (None, None)
        for arxiv_id, pages in self.document_cache.items():
            for page_num, page_text in enumerate(pages):
                ratio = SequenceMatcher(None, chunk, page_text).ratio()
                if ratio > max_ratio:
                    max_ratio = ratio
                    best_match = (arxiv_id, page_num)
        return best_match

    def build_visual_index(self):
        logger.info("Building visual index using ColQwen")
        try:
            pdf_dir = os.path.join(self.data_dir, "docs")
            output_dir = os.path.join(self.data_dir, "visual_embeddings")
            os.makedirs(output_dir, exist_ok=True)
            unique_docs = set()
            for _, row in self.df.iterrows():
                try:
                    unique_docs.update(eval(row['documents']) if 'documents' in row else [])
                except:
                    pass
                if 'doc_path' in row and isinstance(row['doc_path'], str) and row['doc_path'].strip():
                    unique_docs.add(os.path.basename(row['doc_path']))
            pdf_files = [f"{d}.pdf" if not d.endswith('.pdf') else d for d in unique_docs]
            page_embeddings = {}
            for pdf_file in tqdm(pdf_files, desc="Processing PDFs for visual index"):
                doc_id = os.path.splitext(pdf_file)[0]
                pdf_path = os.path.join(pdf_dir, pdf_file)
                if not os.path.exists(pdf_path): continue
                try:
                    pages = convert_from_path(pdf_path)
                except:
                    continue
                for page_idx, page_img in enumerate(pages):
                    page_id = f"{doc_id}_{page_idx}"
                    try:
                        processed_image = self.vision_processor.process_images([page_img])
                        processed_image = {k: v.to(self.vision_model.device) for k, v in processed_image.items()}
                        with torch.no_grad():
                            embedding = self.vision_model(**processed_image)
                        torch.save(embedding.cpu(), os.path.join(output_dir, f"{page_id}.pt"))
                        page_embeddings[page_id] = embedding.cpu()
                    except:
                        continue
            query_embeddings = {}
            for _, row in tqdm(self.df.iterrows(), desc="Processing queries for visual index"):
                q_id = row['q_id']
                try:
                    processed_query = self.vision_processor.process_queries([row['question']])
                    processed_query = {k: v.to(self.vision_model.device) for k, v in processed_query.items()}
                    with torch.no_grad():
                        embedding = self.vision_model(**processed_query)
                    query_embeddings[q_id] = embedding.cpu()
                    torch.save(embedding.cpu(), os.path.join(output_dir, f"query_{q_id}.pt"))
                except:
                    pass
            results = []
            for q_id, query_emb in tqdm(query_embeddings.items(), desc="Ranking documents for queries"):
                try:
                    document_info = self.df[self.df['q_id'] == q_id].iloc[0]
                    question = document_info['question']
                    try:
                        relevant_docs = [doc.split(".pdf")[0] for doc in eval(document_info['documents'])] if 'documents' in document_info else []
                    except:
                        relevant_docs = []
                    if not relevant_docs:
                        relevant_docs = [os.path.splitext(f)[0] for f in pdf_files]
                    relevant_page_embeddings = {pid: emb for pid, emb in page_embeddings.items() if pid.rsplit('_', 1)[0] in relevant_docs}
                    doc_ids, doc_embeddings = list(relevant_page_embeddings.keys()), list(relevant_page_embeddings.values())
                    if not doc_embeddings: continue
                    scores_list = []
                    for doc_emb in doc_embeddings:
                        try:
                            scores_list.append(self.vision_processor.score_multi_vector(query_emb, doc_emb).item())
                        except:
                            scores_list.append(-999.0) 
                    scores = np.array(scores_list)
                    top_indices = np.argsort(-scores)
                    ranked_docs = np.array(doc_ids)[top_indices]
                    for doc_id, score in zip(ranked_docs, scores[top_indices]):
                        results.append({'q_id': q_id, 'document_id': doc_id, 'score': float(score), 'question': question})
                except:
                    continue
            with open(self.vision_retrieval_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['q_id', 'document_id', 'score', 'question'])
                writer.writeheader()
                writer.writerows(results)
            return True
        except Exception:
            return False

    def build_text_index(self):
        logger.info("Building text index using BGE")
        try:
            if not self.document_cache: self.cache_documents()
            all_chunks, chunk_to_doc_mapping = [], []
            for doc_id, pages in tqdm(self.document_cache.items(), desc="Indexing text chunks"):
                for page_idx, page_text in enumerate(pages):
                    if not page_text.strip(): continue
                    for chunk in self.split_text(page_text):
                        all_chunks.append(chunk)
                        chunk_to_doc_mapping.append({'chunk': chunk, 'chunk_pdf_name': doc_id, 'pdf_page_number': page_idx})
            chroma_client = chromadb.Client()
            collection = chroma_client.create_collection(f"st_col_{uuid.uuid4().hex[:8]}", embedding_function=self.st_embedding_function, metadata={"hnsw:space": "cosine"})
            batch_limit = 5000
            total_chunks = len(all_chunks)
            ids = [f"chunk_{i}" for i in range(total_chunks)]
            logger.info(f"Adding {total_chunks} chunks to ChromaDB in batches of {batch_limit}...")
            for i in range(0, total_chunks, batch_limit):
                end = min(i + batch_limit, total_chunks)
                collection.add(documents=all_chunks[i:end], ids=ids[i:end])
            results = []
            for _, row in tqdm(self.df.iterrows(), desc="Processing queries for BGE"):
                q_id, question = row['q_id'], row['question']
                try:
                    query_results = collection.query(query_texts=[question], n_results=self.top_k * 4)
                    for rank, (chunk_idx, score) in enumerate(zip([int(id.split('_')[1]) for id in query_results['ids'][0]], query_results['distances'][0])):
                        chunk_info = chunk_to_doc_mapping[chunk_idx]
                        results.append({
                            'q_id': q_id, 'question': question, 'chunk': all_chunks[chunk_idx],
                            'chunk_pdf_name': chunk_info['chunk_pdf_name'], 'pdf_page_number': chunk_info['pdf_page_number'],
                            'rank': rank + 1, 'score': 1.0 - score 
                        })
                except:
                    continue
            with open(self.text_retrieval_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['q_id', 'question', 'chunk', 'chunk_pdf_name', 'pdf_page_number', 'rank', 'score'])
                writer.writeheader()
                writer.writerows(results)
            return True
        except Exception:
            return False

    def retrieve_visual_contexts(self, query_id, query_text=""):
        return self.visual_agent.retrieve(query_id, query_text)

    def retrieve_textual_contexts(self, query_id, query_text=""):
        return self.text_agent.retrieve(query_id, query_text)

    def _apply_adaptive_soft_compression(self, image, is_high_res):
        import math
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB") if isinstance(image, str) else image
        width, height = image.size
        factor = 28 
        if is_high_res:
            target_max_pixels, target_min_pixels = factor * factor * 1280, factor * factor * 4
        else:
            target_max_pixels, target_min_pixels = factor * factor * 64, factor * factor * 4
        h_bar, w_bar = round(height / factor) * factor, round(width / factor) * factor
        if h_bar * w_bar > target_max_pixels:
            beta = math.sqrt((height * width) / target_max_pixels)
            h_bar = max(factor, math.floor(height / beta / factor) * factor)
            w_bar = max(factor, math.floor(width / beta / factor) * factor)
        elif h_bar * w_bar < target_min_pixels:
            beta = math.sqrt(target_min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        if (width, height) != (w_bar, h_bar):
            image = image.resize((w_bar, h_bar), Image.Resampling.LANCZOS)
        return image

    def generate_combined_logic(self, prompt):
        """裁判专用：独立于专家任务的 LLM 调用接口 (纯双核适配)"""
        token_stats = {"input_tokens": 0, "output_tokens": 0}
        try:
            if self.llm_model == "internvl3":
                resp = self.client.chat.completions.create(model="internvl3", messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}], temperature=0.0)
                token_stats = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
                return resp.choices[0].message.content, token_stats
            elif self.llm_model == "qwen":
                messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.qwen_processor(text=[text], padding=True, return_tensors="pt").to("cuda")
                token_stats["input_tokens"] = inputs.input_ids.shape[1]
                generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=1024)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                token_stats["output_tokens"] = len(generated_ids_trimmed[0])
                return self.qwen_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0], token_stats
        except Exception as e:
            logger.error(f"Referee arbitration failed: {e}")
            return "Arbitration Error", {"input_tokens": 0, "output_tokens": 0}

    def generate_visual_response(self, query, visual_contexts, custom_prompt=None):
        try:
            token_stats = {"input_tokens": 0, "output_tokens": 0}
            prompt_template = custom_prompt.replace("{query}", str(query)).replace("{context}", "Images provided.") if custom_prompt else self.current_prompts['visual'].format(query=query)
            if self.llm_model == "internvl3":
                content_list = []
                for ctx in (visual_contexts or []):
                    processed_img = self._apply_adaptive_soft_compression(ctx['image'], ctx.get('high_res', True))
                    buf = BytesIO(); processed_img.save(buf, format="JPEG")
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                content_list.append({"type": "text", "text": prompt_template})
                resp = self.client.chat.completions.create(model="internvl3", messages=[{"role": "user", "content": content_list}], temperature=0.0, max_tokens=1024)
                token_stats = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
                raw_output = resp.choices[0].message.content
            elif self.llm_model == "qwen":
                image_inputs_config = [{"type": "image", "image": self._apply_adaptive_soft_compression(ctx['image'], ctx.get('high_res', True))} for ctx in (visual_contexts or [])]
                messages = [{"role": "user", "content": image_inputs_config + [{"type": "text", "text": prompt_template}]}]
                text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = self.process_vision_info(messages)
                inputs = self.qwen_processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")
                token_stats["input_tokens"] = inputs.input_ids.shape[1]
                generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=1024)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                token_stats["output_tokens"] = len(generated_ids_trimmed[0])
                raw_output = self.qwen_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
            logger.info(f"📊 [Token] {self.llm_model} Visual: In={token_stats['input_tokens']}, Out={token_stats['output_tokens']}")
            return raw_output, token_stats
        except Exception as e:
            logger.error(f"Error generating visual response: {str(e)}")
            return "Error generating response from visual contexts.", {"input_tokens": 0, "output_tokens": 0}

    def generate_textual_response(self, query, textual_contexts, custom_prompt=None):
        try:
            contexts_str = "\n- ".join([ctx.get('chunk') or ctx.get('content') or "" for ctx in textual_contexts])
            token_stats = {"input_tokens": 0, "output_tokens": 0}
            if custom_prompt:
                prompt_template = custom_prompt.format(query=query, context=contexts_str) if "{context}" in custom_prompt else f"{custom_prompt}\nQuestion: {query}\nContext: {contexts_str}"
            else:
                prompt_template = self.current_prompts['text'].format(query=query, context=contexts_str)
            if self.llm_model == "internvl3":
                resp = self.client.chat.completions.create(model="internvl3", messages=[{"role": "user", "content": [{"type": "text", "text": prompt_template}]}], temperature=0.0)
                token_stats = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
                raw_output = resp.choices[0].message.content
            elif self.llm_model == "qwen":
                messages = [{"role": "user", "content": [{"type": "text", "text": prompt_template}]}]
                text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.qwen_processor(text=[text], padding=True, return_tensors="pt").to("cuda")
                token_stats["input_tokens"] = inputs.input_ids.shape[1]
                generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=1024)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                token_stats["output_tokens"] = len(generated_ids_trimmed[0])
                raw_output = self.qwen_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
            logger.info(f"📊 [Token] {self.llm_model} Text: In={token_stats['input_tokens']}, Out={token_stats['output_tokens']}")
            return raw_output, token_stats
        except Exception as e:
            logger.error(f"Error generating text response: {str(e)}")
            return "Error generating response from text contexts.", {"input_tokens": 0, "output_tokens": 0}

    def predict_route(self, question):
        if not self.use_router: return "fusion"
        try:
            return self.router.predict_modality(question)
        except Exception as e:
            logger.error(f"Router Exception: {e}. Falling back to fusion.")
            return "fusion"

    def extract_sections(self, text):
        sections = {"Evidence": "", "Chain of Thought": "", "Answer": ""}
        if not text or not text.strip(): return sections
        found_any_tag = False
        for heading in ["Evidence", "Chain of Thought", "Answer"]:
            match = re.search(rf"(?:##\s*)?{heading}\s*[:：]\s*(.*?)(?=(?:##\s*)?(?:Evidence|Chain of Thought|Answer)\s*[:：]|$)", text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                for p in ["(extract data here)", "(analyze here)", "(final sentence here)", "[Data found]"]:
                    content = content.replace(p, "")
                sections[heading] = content.strip()
                found_any_tag = True
        if not found_any_tag or not sections["Answer"]:
            clean_answer = text
            for label in ["## Evidence:", "## Chain of Thought:", "## Answer:", "Evidence:", "Chain of Thought:", "Answer:"]:
                clean_answer = clean_answer.replace(label, "")
            sections["Answer"] = clean_answer.strip()
            sections["Evidence"] = "Direct output (No tags detected)"
        return sections

    def parse_combined_output(self, output):
        sections = {'Analysis': '', 'Conclusion': '', 'Final Answer': ''}
        current_section = None
        for line in output.split('\n'):
            if line.startswith('## '): current_section = line[3:].strip(':')
            elif current_section and current_section in sections: sections[current_section] += line + '\n'
        for key in sections: sections[key] = sections[key].strip()
        return sections

    def process_query(self, query_id, custom_prompt=None):
        try:
            query_id_str = str(query_id).strip()
            q_id_clean = re.sub(r'\.0$', '', query_id_str)
            safe_id = query_id_str.replace('/', '$')
            combined_file = f"{self.output_dir}/{self.llm_model}_apdcrag/response_{safe_id}.json"

            if os.path.exists(combined_file) and not self.overwrite_results:
                return True

            time_stats = {"router_time": 0.0, "text_time": 0.0, "vision_time": 0.0, "fusion_time": 0.0, "total_time": 0.0}
            total_token_stats = {"text_tokens": {"in": 0, "out": 0}, "vision_tokens": {"in": 0, "out": 0}, "fusion_tokens": {"in": 0, "out": 0}, "grand_total": 0}
            text_chunks_count, visual_pages_count = 0, 0
            retrieved_text_chunks = []
            start_total = time.time()

            filtered_df = self.df[self.df['q_id'].astype(str).str.replace(r'\.0$', '', regex=True) == q_id_clean]
            if filtered_df.empty:
                logger.error(f"Query ID {q_id_clean} not found!")
                return False

            query_row = filtered_df.iloc[0]
            question = query_row['question']
            try:
                raw_ans = query_row['answer']
                answer = eval(raw_ans) if isinstance(raw_ans, str) else raw_ans
            except:
                answer = query_row['answer']

            t_router_start = time.time()
            route = self.router.predict_modality(question) if self.use_router else "fusion"
            time_stats["router_time"] = round(time.time() - t_router_start, 4)

            def get_text_response():
                nonlocal text_chunks_count, retrieved_text_chunks
                t_start = time.time()
                try:
                    contexts = self.retrieve_textual_contexts(query_id_str)
                    if not contexts: return None, 0.0, {"in": 0, "out": 0}
                    text_chunks_count = len(contexts)
                    retrieved_text_chunks = [c.get('content') or c.get('chunk') or "" for c in contexts]
                    raw, tokens = self.generate_textual_response(question, contexts, custom_prompt=custom_prompt)
                    res = self.extract_sections(raw); res['raw_output'] = raw
                    return res, round(time.time() - t_start, 3), tokens
                except Exception as e:
                    logger.error(f"Text Branch Error: {e}")
                    return None, round(time.time() - t_start, 3), {"in": 0, "out": 0}

            def get_visual_response():
                nonlocal visual_pages_count
                t_start = time.time()
                try:
                    contexts = self.retrieve_visual_contexts(query_id_str)
                    if not contexts: return None, 0.0, {"in": 0, "out": 0}
                    visual_pages_count = len(contexts)
                    raw, tokens = self.generate_visual_response(question, contexts, custom_prompt=custom_prompt)
                    res = self.extract_sections(raw); res['raw_output'] = raw
                    return res, round(time.time() - t_start, 3), tokens
                except Exception as e:
                    logger.error(f"Visual Branch Error: {e}")
                    return None, round(time.time() - t_start, 3), {"in": 0, "out": 0}

            def local_is_valid(resp_dict):
                if not resp_dict or "Answer" not in resp_dict or not resp_dict["Answer"]: return False
                ans = str(resp_dict["Answer"]).strip().lower()
                bad_keywords = ["not found", "no information", "cannot", "n/a", "unknown", "error", "insufficient", "not provided","not provided."]
                return not any(b in ans for b in bad_keywords)

            final_response_obj = {}
            text_res, vis_res = None, None
            current_route = route if self.use_router else "fusion"
            fallback_chain = [current_route]

            if current_route == "text":
                text_res, time_stats["text_time"], total_token_stats["text_tokens"] = get_text_response()
                if local_is_valid(text_res): final_response_obj = text_res
                else:
                    logger.info(f"⏩ [QID {q_id_clean}] Text branch returned 'Not provided'. Falling back to Visual...")
                    current_route = "visual"; fallback_chain.append("visual")

            if current_route == "visual":
                vis_res, time_stats["vision_time"], total_token_stats["vision_tokens"] = get_visual_response()
                if local_is_valid(vis_res): final_response_obj = vis_res
                else:
                    logger.info(f"⏩ [QID {q_id_clean}] Visual branch returned 'Not provided'. Falling back to Fusion...")
                    current_route = "fusion"; fallback_chain.append("fusion")

            if current_route == "fusion":
                if not text_res: text_res, time_stats["text_time"], total_token_stats["text_tokens"] = get_text_response()
                if not vis_res: vis_res, time_stats["vision_time"], total_token_stats["vision_tokens"] = get_visual_response()
                if self.use_decision:
                    f_s = time.time()
                    f_obj, tok = self.decision_agent.collaborate(question, text_res, vis_res)
                    total_token_stats["fusion_tokens"] = tok
                    combined_ans = f_obj.get("Final Answer") or f_obj.get("Answer")
                    final_response_obj = {"Answer": combined_ans if combined_ans else "Synthesis failed."}
                    time_stats["fusion_time"] = round(time.time() - f_s, 3)
                else:
                    v_ans = vis_res.get("Answer", "Visual Not Found") if vis_res else "Visual Error"
                    t_ans = text_res.get("Answer", "Text Not Found") if text_res else "Text Error"
                    final_response_obj = {"Answer": f"Visual: {v_ans}\nText: {t_ans}"}

            used_method = f"router_{fallback_chain[0]}" if len(fallback_chain) == 1 else f"fallback_{'_to_'.join(fallback_chain)}"

            def sum_tokens(stat_dict): 
                return stat_dict.get("input_tokens", stat_dict.get("in", 0)) + stat_dict.get("output_tokens", stat_dict.get("out", 0))
            
            cost_text = sum_tokens(total_token_stats["text_tokens"])
            cost_vision = sum_tokens(total_token_stats["vision_tokens"])
            cost_fusion = cost_text + cost_vision + sum_tokens(total_token_stats["fusion_tokens"])
            
            if "router_visual" in used_method:
                total_token_stats["grand_total"] = cost_vision
                self.track_visual_tokens.append(cost_vision)
            elif "router_text" in used_method:
                total_token_stats["grand_total"] = cost_text
            else:
                total_token_stats["grand_total"] = cost_fusion
                self.track_fusion_tokens.append(cost_fusion)

            time_stats["total_time"] = round(time.time() - start_total, 4)
            final_ans_str = final_response_obj.get("Answer", "No Answer Generated")

            final_output = {
                "q_id": query_id_str, "question": question, "gt_answer": answer,
                "router_prediction": route, "actual_method": used_method,
                "time_consumption": time_stats, 
                "token_consumption": total_token_stats,
                "answer": final_ans_str, 
                "retrieved_context_chunks": retrieved_text_chunks,
                "visual_raw": vis_res.get("raw_output", "") if vis_res else "Skipped",
                "text_raw": text_res.get("raw_output", "") if text_res else "Skipped",
                "gmm_info": {"text_chunks": text_chunks_count, "visual_pages": visual_pages_count},
                "ablation_config": {
                    "use_router": self.use_router, "use_gmm": self.use_gmm, "use_decision": self.use_decision
                }
            }

            os.makedirs(os.path.dirname(combined_file), exist_ok=True)
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, indent=4, ensure_ascii=False)
            
            self._record_router_debug(final_output)

            print("\n" + "─"*60)
            print(f"📌 [QID]: {q_id_clean} | [Route]: {used_method} | [Time]: {time_stats['total_time']:.2f}s")
            print(f"✅ [GT Answer]: {answer}")
            print(f"🤖 [Model Pred]: {final_ans_str}")
            print("─"*60)

            return True

        except Exception as e:
            logger.error(f"Error processing ID {query_id}: {str(e)}")
            traceback.print_exc()
            return False

    def _record_router_debug(self, data):
        """记录详细的决策、时间与 Token 统计到 CSV"""
        csv_path = f"{self.output_dir}/innovation_performance_report.csv"
        file_exists = os.path.isfile(csv_path)
        t = data['time_consumption']
        tok = data['token_consumption']

        def get_tok(branch):
            b = tok.get(branch, {})
            return b.get("in", b.get("input_tokens", 0)) + b.get("out", b.get("output_tokens", 0))
            
        with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
            fieldnames = [
                'q_id', 'router_prediction', 'actual_method', 
                't_router', 't_text', 't_vision', 't_fusion', 't_total',
                'tok_text', 'tok_vision', 'tok_fusion', 'tok_total',
                'answer', 'gt_answer'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists: writer.writeheader()
            
            writer.writerow({
                'q_id': data['q_id'],
                'router_prediction': data['router_prediction'],
                'actual_method': data['actual_method'],
                't_router': t.get('router_time', 0),
                't_text': t.get('text_time', 0),
                't_vision': t.get('vision_time', 0),
                't_fusion': t.get('fusion_time', 0),
                't_total': t.get('total_time', 0),
                'tok_text': get_tok("text_tokens"),
                'tok_vision': get_tok("vision_tokens"),
                'tok_fusion': get_tok("fusion_tokens"),   
                'tok_total': tok.get('grand_total', 0),   
                'answer': str(data['answer'])[:100].replace('\n', ' '),
                'gt_answer': str(data['gt_answer'])[:100].replace('\n', ' ')
            })

    def run(self):
        """Run the APDC-RAG pipeline on all queries in the dataset."""
        print("\n" + "="*50)
        print("      APDC-RAG Pipeline Configuration")
        print("="*50)
        print(f"  > Router (创新点1):    {'[ ON ]' if getattr(self, 'use_router', False) else '[ OFF ]'}")
        print(f"  > GMM Filter (创新点2): {'[ ON ]' if getattr(self, 'use_gmm', False) else '[ OFF ]'}")
        print(f"  > Decision (创新点3):    {'[ ON ]' if getattr(self, 'use_decision', False) else '[ OFF ]'}")
        print(f"  > LLM Model:            {self.llm_model}")
        print("="*50 + "\n")

        logger.info("Starting APDCRAG pipeline")
        
        for query_id in tqdm(self.df['q_id'].unique()):
            try:
                success = self.process_query(query_id)
                if not success:
                    logger.warning(f"Failed to process query {query_id}")
            except Exception as e:
                logger.error(f"Error processing query {query_id}: {str(e)}")
        
        logger.info("APDC-RAG pipeline completed")

        avg_vis = np.mean(self.track_visual_tokens) if self.track_visual_tokens else 0
        avg_fus = np.mean(self.track_fusion_tokens) if self.track_fusion_tokens else 0
        
        print("\n" + "="*55)
        print(" 📊 专项 Token 消耗分析报表 (Ablation Stats)")
        print("="*55)
        print(f" [纯视觉路径 (Visual)] 样本数: {len(self.track_visual_tokens)}, 平均 Token: {avg_vis:.1f}")
        print(f" [图文融合路径 (Fusion)] 样本数: {len(self.track_fusion_tokens)}, 平均 Token: {avg_fus:.1f}")
        print("="*55 + "\n")