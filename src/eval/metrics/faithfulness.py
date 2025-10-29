from src.eval.metrics.base_metric import BaseMetric
from sentence_transformers import util

class Faithfulness(BaseMetric):
    def compute(self, row, group=None):
        pred = row.get("pred", "")
        contexts = row.get("contexts", [])
        if not contexts:
            return 0.0
        emb_pred = self.model.encode(pred, convert_to_tensor=True, show_progress_bar=False)
        scores = [
            util.cos_sim(emb_pred, self.model.encode(ctx, convert_to_tensor=True, show_progress_bar=False)).item()
            for ctx in contexts
        ]
        return max(scores)

