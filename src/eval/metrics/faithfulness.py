from src.eval.metrics.base_metric import BaseMetric
from sentence_transformers import util

class Faithfulness(BaseMetric):
    """
    Compute the Faithfulness metric.
    Measures how well the generated answer semantically aligns with the provided contexts
    using cosine similarity between embeddings.
    """
    def compute(self, row, group=None):
        """
        Args:
            row (dict): A single result containing 'pred' (model output) and 'contexts' (retrieved passages).
            group (list, optional): Unused here, included for interface compatibility.

        Returns:
            float: Maximum cosine similarity between the prediction and its contexts.
        """
        pred = row.get("pred", "")
        contexts = row.get("contexts", [])

        # Return 0 if no context is available for comparison.
        if not contexts:
            return 0.0
        emb_pred = self.model.encode(pred, convert_to_tensor=True, show_progress_bar=False)
        scores = [
            util.cos_sim(emb_pred, self.model.encode(ctx, convert_to_tensor=True, show_progress_bar=False)).item()
            for ctx in contexts
        ]
        return max(scores)

