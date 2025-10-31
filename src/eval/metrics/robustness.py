from src.eval.metrics.base_metric import BaseMetric
from sentence_transformers import util
import numpy as np

class RobustnessByDegree(BaseMetric):
    """
    Compute the Robustness metric based on answer consistency across variations.
    Measures the average semantic similarity between all predicted answers
    within the same question group.
    """

    def compute(self, row, group=None):
        """
        Args:
            row (dict): Unused; kept for interface compatibility.
            group (list): List of results (same question with variations in phrasing or degree).

        Returns:
            float or None: Average pairwise cosine similarity between predictions,
                           or None if robustness cannot be computed (group < 2).
        """
        if not group or len(group) < 2:
            return None  # Pas de robustesse calculable
        embs = [self.model.encode(r["pred"], convert_to_tensor=True, show_progress_bar=False) for r in group]
        sims = []
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                sims.append(util.cos_sim(embs[i], embs[j]).item())
        return float(np.mean(sims))

