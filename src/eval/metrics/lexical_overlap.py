from src.eval.metrics.base_metric import BaseMetric

class LexicalOverlap(BaseMetric):
    """
    Compute the Lexical Overlap metric.
    Measures the proportion of words from the ground-truth answer
    that also appear in the model prediction.
    """
    def compute(self, row, group=None):
        """
        Args:
            row (dict): A single result containing 'pred' and 'answer' fields.
            group (list, optional): Unused here, included for interface compatibility.

        Returns:
            float: Ratio of overlapping words between prediction and ground truth.
        """
        pred, gold = row.get("pred", ""), row.get("answer", "")
        set_pred = set(pred.lower().split())
        set_gold = set(gold.lower().split())

        # Return 0 if the reference answer is empty to avoid division by zero.
        return len(set_pred & set_gold) / len(set_gold) if set_gold else 0.0
