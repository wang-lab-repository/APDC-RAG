import numpy as np
from sklearn.mixture import GaussianMixture
import logging

logger = logging.getLogger("HybridRetrieval")

class HybridAdaptiveRetrieval:
    def __init__(self):
        self.max_k = 5             
        self.obs_pool = 15         

    def get_resolution_boundary(self, scores, mode="visual"):
        if mode == "text": return self.max_k
        S_scores = np.array(scores, dtype=np.float64)
        candidate_scores = S_scores[:self.obs_pool]

        try:
            # =========================================================
            # 1. 微观层面
            # =========================================================
            S_top = candidate_scores[:5]
            delta_S = S_top[:-1] - S_top[1:]      
            delta_S_max = np.max(delta_S)         
            I_max = np.argmax(delta_S)            
            avg_delta = np.mean(delta_S) + 1e-6

            # =========================================================
            # 2. 宏观层面
            # =========================================================
            # 采用 tied 协方差矩阵，在小样本得分分布下更稳定
            gmm = GaussianMixture(n_components=2, covariance_type='tied', random_state=42, max_iter=20)
            X = candidate_scores.reshape(-1, 1)
            gmm.fit(X)

            # =========================================================
            # 3. 宏观不确定性量化：后验概率 P 与 平均香农熵 H
            # =========================================================
            # 获取 Top-5 样本属于各个簇的后验概率矩阵
            proba = gmm.predict_proba(S_top.reshape(-1, 1))
            
            # 推导出平均香农熵 H 作为资源调度的核心调节因子
            H_entropy = -np.sum(proba * np.log(proba + 1e-9)) / len(proba)

            # =========================================================
            # 4. IE-GMM 动态决策引擎 (Entropy-Aware Thresholding)
            # =========================================================
            # 利用宏观信息熵 H 动态调节梯度触发阈值
            dynamic_multiplier = 2.0 + (H_entropy * 0.6) 
            dynamic_multiplier = np.clip(dynamic_multiplier, 1.9, 2.2)

            # 融合决策：结合局部得分梯度共同指导动态截断边界
            if delta_S_max > 0.5 and (delta_S_max > dynamic_multiplier * avg_delta):
                # 截断点定位：基于断崖点 I 进行资源边界划分 K
                K_boundary = int(np.clip(I_max + 2, 1, 5))
                
                logger.info(f"[IE-GMM] 熵 H:{H_entropy:.2f} 调节倍数:{dynamic_multiplier:.2f} | 梯度 \\Delta S_max:{delta_S_max:.2f} -> 截断边界 K:{K_boundary}")
                return K_boundary

            # 熵值较高：说明得分分布平缓，采用保守策略
            logger.info(f"[IE-GMM] 高熵保守策略 (H:{H_entropy:.2f}) -> 全量 {self.max_k} 页保留高清")
            return self.max_k

        except Exception as e:
            logger.warning(f"IE-GMM 算法降级: {e}")
            return self.max_k

    def dual_scale_dynamic_k(self, scores, mode="text"):
        return self.max_k

    def retrieve(self, items, scores, mode="text"):
        if not items:
            return []
        k = self.dual_scale_dynamic_k(scores, mode=mode)
        return items[:k]

    def adaptive_visual(self, items, scores):
        return self.retrieve(items, scores, mode="visual")