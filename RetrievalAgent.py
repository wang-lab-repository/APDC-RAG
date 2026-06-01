import pandas as pd
import os
import re
import numpy as np
import logging
import traceback
from pdf2image import convert_from_path
from hybrid_retrieval import HybridAdaptiveRetrieval  

logger = logging.getLogger("Retrieval_Agents")

# =========================================================
# 基类：RetrievalAgentBase (通用检索调度层)
# =========================================================
class RetrievalAgentBase:
    def __init__(self, data_dir, retrieval_file, pool_size=20, agent_name="BaseAgent", use_gmm=True, build_func=None):
        self.data_dir = data_dir
        self.retrieval_file = retrieval_file
        self.pool_size = pool_size
        self.agent_name = agent_name
        self.use_gmm = use_gmm
        self.adaptive = HybridAdaptiveRetrieval()
        self.build_func = build_func  # 接收离线索引构建函数

    def _normalize_id(self, df):
        """标准化 Query ID 格式，解决不同评测集的数据类型偏差"""
        if 'q_id' in df.columns:
            return df['q_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        return pd.Series()

    def load_candidates(self, query_id):
        """加载稠密检索阶段召回的候选池 (Candidate Pool)"""
        if not os.path.exists(self.retrieval_file):
            logger.warning(f"[{self.agent_name}] 索引文件缺失: {self.retrieval_file}")
            if self.build_func:
                logger.info(f"[{self.agent_name}] 触发自动索引构建流水线...")
                success = self.build_func() 
                if not success or not os.path.exists(self.retrieval_file):
                    logger.error(f"[{self.agent_name}] 索引构建失败！")
                    return [], []
            else:
                logger.error(f"[{self.agent_name}] 未配置构建函数，无法初始化检索池。")
                return [], []
        try:
            df = pd.read_csv(self.retrieval_file)
            target_id = str(query_id).strip().replace('.0', '')
            df['q_id_norm'] = self._normalize_id(df)
            
            query_rows = df[df['q_id_norm'] == target_id]

            if len(query_rows) == 0:
                logger.warning(f"[{self.agent_name}] 未找到匹配 ID: {target_id}")
                return [], []

            # 确定排序依据：优先依据特征相似度 score，降级使用 rank
            if 'score' in query_rows.columns:
                candidates = query_rows.sort_values(by="score", ascending=False).head(self.pool_size)
                scores = candidates['score'].tolist()
            else:
                candidates = query_rows.sort_values('rank', ascending=True).head(self.pool_size)
                scores = [1.0 / (r + 1) for r in range(len(candidates))]

            items = candidates.to_dict('records')
            return items, scores

        except Exception as e:
            logger.error(f"[{self.agent_name}] 加载候选失败: {e}")
            return [], []

    def format_results(self, items):
        """
        格式对齐层：
        确保返回的上下文序列包含标准的 'content' 字段，支撑下游 apdcrag.py 的生成器调用。
        """
        results = []
        if not items:
            return []
            
        for it in items:
            if isinstance(it, dict):
                new_item = it.copy()
                if 'content' not in new_item:
                    new_item['content'] = it.get('chunk') or it.get('text') or ""
                results.append(new_item)
            else:
                results.append({
                    'content': str(it),
                    'chunk': str(it)
                })
                
        return results

# =========================================================
# 文本逻辑检索专家：OCRTextAgent
# =========================================================
class OCRTextAgent(RetrievalAgentBase):
    def __init__(self, retrieval_file, data_dir=None, pool_size=20, use_gmm=True, build_func=None): 
        super().__init__(data_dir, retrieval_file, pool_size, agent_name="OCR-Text-Agent", use_gmm=use_gmm, build_func=build_func)        
        
    def retrieve(self, query_id, query=""):
        items, scores = self.load_candidates(query_id)
        if not items:
            return []

        top_20_scores = [round(float(s), 4) for s in scores[:20]]
        logger.info(f"📊 QID {query_id} [Textual] 初始候选得分分布: {top_20_scores}")

        if self.use_gmm:
            final_items = self.adaptive.retrieve(items, scores, mode="text")
        else:
            final_items = items[:5]
            logger.info("[Textual] Ablation: Adaptive context sizing disabled -> Fixed Top-5")
            
        logger.info(f"[Textual] 最终入参文本块数量: {len(final_items)}")
        return self.format_results(final_items)

# =========================================================
# 视觉感知检索专家：ColQwenVisualAgent 
# (融合 IE-GMM 软压缩标记)
# =========================================================
class ColQwenVisualAgent(RetrievalAgentBase):
    def __init__(self, data_dir, retrieval_file, pool_size=20, use_gmm=True, build_func=None):
        super().__init__(data_dir, retrieval_file, pool_size, agent_name="Visual-Agent", use_gmm=use_gmm, build_func=build_func)

    def retrieve(self, query_id, query=""):
        items, scores = self.load_candidates(query_id)
        if not items: return []
        
        selected_items = items[:5] 
        
        # ==========================================
        # 消融实验开关：控制是否启用 IE-GMM 动态分辨率 (软压缩)
        # ==========================================
        if self.use_gmm:
            high_res_k = self.adaptive.get_resolution_boundary(scores, mode="visual")
        else:
            # GMM 降级 (Baseline)：所有候选页面均维持全量高分辨率特征
            high_res_k = 5
            logger.info("[Visual] Ablation: IE-GMM disabled -> Full resolution maintained")

        pages = []
        pdf_dir = os.path.join(self.data_dir, "docs")
        
        for idx, item in enumerate(selected_items):
            doc_id = str(item['document_id'])
            if '_' not in doc_id: continue
            base_doc, p_num_str = doc_id.rsplit('_', 1)
            try:
                page_idx = int(p_num_str)
                pdf_path = os.path.join(pdf_dir, f"{base_doc}.pdf")
                if not os.path.exists(pdf_path): continue
                
                imgs = convert_from_path(pdf_path, first_page=page_idx+1, last_page=page_idx+1, dpi=150)
                if imgs:
                    pages.append({
                        'image': imgs[0],
                        'document_id': doc_id,
                        'page_number': page_idx,
                        'high_res': idx < high_res_k  
                    })
            except Exception as e:
                logger.error(f"  [Visual] 跨模态上下文加载失败 {doc_id}: {e}")

        logger.info(f"📸 QID {query_id} 页面级视觉召回完成 -> 核心高分页面:{high_res_k}页 | 软压缩边缘页面:{len(pages)-high_res_k}页")
        return pages