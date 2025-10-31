from src.eval.metrics.base_metric import BaseMetric
from sentence_transformers import util

class SemanticSimilarity(BaseMetric):
    """
    Compute the Semantic Similarity metric.
    Measures the cosine similarity between embeddings of the predicted and reference answers.
    """
    def compute(self, row, group=None):
        """
        Args:
            row (dict): A single result containing 'pred' (model output) and 'answer' (reference).
            group (list, optional): Unused here, included for interface compatibility.

        Returns:
            float: Cosine similarity between predicted and reference embeddings.
        """
        pred, gold = row.get("pred", ""), row.get("answer", "")
        emb1 = self.model.encode(pred, convert_to_tensor=True, show_progress_bar=False)
        emb2 = self.model.encode(gold, convert_to_tensor=True, show_progress_bar=False)
        return util.cos_sim(emb1, emb2).item()