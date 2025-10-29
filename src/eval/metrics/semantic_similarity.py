from src.eval.metrics.base_metric import BaseMetric
from sentence_transformers import util

class SemanticSimilarity(BaseMetric):
    def compute(self, row, group=None):
        pred, gold = row.get("pred", ""), row.get("answer", "")
        emb1 = self.model.encode(pred, convert_to_tensor=True, show_progress_bar=False)
        emb2 = self.model.encode(gold, convert_to_tensor=True, show_progress_bar=False)
        return util.cos_sim(emb1, emb2).item()