import logging
import re
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
logger = logging.getLogger("DecisionAgent")

class HierarchicalDecisionAgent:
    def __init__(self, llm_gateway, use_decision=True):

        self.gateway = llm_gateway
        self.use_decision = use_decision
        
        if self.use_decision:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def _calculate_alignment(self, text1, text2):

        if not text1 or not text2: 
            return 0.0

        lexical_score = SequenceMatcher(None, text1, text2).ratio()

        emb1 = self.embedder.encode([text1])
        emb2 = self.embedder.encode([text2])
        semantic_score = cosine_similarity(emb1, emb2)[0][0]


        S_align = (0.3 * lexical_score) + (0.7 * semantic_score)

        S_align = max(0.0, min(1.0, float(S_align)))

        logger.info(f"[Alignment Matrix] Lexical: {lexical_score:.2f} | Semantic: {semantic_score:.2f} -> S_align: {S_align:.2f}")
        return S_align

    def run_textual_expert(self, query, contexts):
        """
        文本逻辑专家：利用 gateway 生成纯文本维度的 CoT
        """
        logger.info("启动 [文本逻辑专家]...")
        # 直接调用主类的生成方法，但可以传入专家特有的 prompt 或参数
        raw_output = self.gateway.generate_textual_response(query, contexts)
        
        # 兼容处理：有些返回是 (text, stats)，有些只是 text
        text = raw_output[0] if isinstance(raw_output, tuple) else raw_output
        return self.gateway.extract_sections(text)

    def run_visual_expert(self, query, visual_contexts):
        """
        视觉感知专家：利用 gateway 生成基于图像的 CoT
        """
        logger.info("启动 [视觉感知专家]...")
        raw_output = self.gateway.generate_visual_response(query, visual_contexts)
        
        text = raw_output[0] if isinstance(raw_output, tuple) else raw_output
        return self.gateway.extract_sections(text)
    
    def collaborate(self, query, text_expert_out, vis_expert_out):


        S_align = self._calculate_alignment(
            text_expert_out.get('Answer', ''), 
            vis_expert_out.get('Answer', '')
        )
        
        # 定义置信度阈值 \tau
        tau_threshold = 0.8 
        
        if not self.use_decision:
            logger.info("Ablation: Referee Protocol OFF. Executing simple concatenation.")
            return {
                "Final Answer": f"{vis_expert_out['Answer']} (Visual) | {text_expert_out['Answer']} (Textual)",
                "Analysis": "Decision logic disabled.",
                "Method": "Simple_Concatenation"
            }

        # =========================================================
        # 2. 达成共识 (S_align > \tau)
        # =========================================================
        if S_align > tau_threshold:
            logger.info(f"S_align ({S_align:.2f}) > tau. Consensus reached.")
            # 若达成共识，默认采用视觉感知专家的输出 (包含更丰富的版面原生线索)
            return {
                "Final Answer": vis_expert_out['Answer'],
                "Analysis": "Textual and Visual experts reached high consensus.",
                "Method": "Consensus_Pass"
            }
        
        # =========================================================
        # 3. 跨模态认知冲突 (S_align <= \tau) -> 触发一致性裁判协议
        # =========================================================
        logger.warning(f"Cognitive conflict detected (S_align: {S_align:.2f}). Invoking Referee Protocol...")
        
        # 注入裁判系统提示词 (Referee System Prompt)
        judge_prompt = f"""
        Analyze the following two independent expert responses to the question: "{query}"

        Response 1 (Visual Perception Expert E_vision):
        Evidence: {vis_expert_out.get('Evidence', "N/A")}
        Chain of Thought: {vis_expert_out.get('Chain of Thought', "N/A")}
        Final Answer: {vis_expert_out['Answer']}

        Response 2 (Textual Logic Expert E_text):
        Evidence: {text_expert_out.get('Evidence', "N/A")}
        Chain of Thought: {text_expert_out.get('Chain of Thought', "N/A")}
        Final Answer: {text_expert_out['Answer']}

        Arbitration Rules:
        1. Response 1 is visual-based; Response 2 is text-based.
        2. Generally, given both responses have logical chains of thoughts, and decision boils down to evidence, you should place a higher degree of trust on the evidence reported in Response 1 (Visual).
        3. If one of the responses has declined giving a clear answer (e.g., "Unknown" or "N/A"), please weigh the other answer more.
        4. Language of the answer should be short and direct, usually answerable in a single sentence, or phrase.

        Consider both chains of thought and final answers. Provide your analysis in the following format:

        ## Analysis:
        [Your detailed analysis here]

        ## Conclusion:
        [Your conclusion on which answer is more likely to be correct]

        ## Final Answer:
        [Answer the question based on your arbitration. Must be a single, concise sentence.]
        """
        
        # 执行决策级答案融合 (Decision-based Answer-level Fusion)
        judge_output_raw = self.gateway.generate_combined_logic(judge_prompt)

        if isinstance(judge_output_raw, tuple):
            raw_text, tokens = judge_output_raw
        else:
            raw_text, tokens = judge_output_raw, {"in": 0, "out": 0}

        refined_result = self.gateway.parse_combined_output(raw_text)
        refined_result["Method"] = "Referee_Arbitration"
        return refined_result, tokens