from src.eval.metrics.base_metric import BaseMetric
from sentence_transformers import util
import numpy as np

class RobustnessByDegree(BaseMetric):
    def compute(self, row, group=None):
        if not group or len(group) < 2:
            return None  # Pas de robustesse calculable
        embs = [self.model.encode(r["pred"], convert_to_tensor=True, show_progress_bar=False) for r in group]
        sims = []
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                sims.append(util.cos_sim(embs[i], embs[j]).item())
        return float(np.mean(sims))

