from src.eval.metrics.base_metric import BaseMetric

class ExactMatch(BaseMetric):
    """
    Compute the Exact Match metric.
    Returns 1.0 if the ground-truth answer is found within the prediction, else 0.0.
    """
    def compute(self, row, group=None):
        """
        Args:
            row (dict): A single result containing 'pred' and 'answer' fields.
            group (list, optional): Unused here, included for interface compatibility.

        Returns:
            float: 1.0 if the gold answer is contained in the prediction, else 0.0.
        """
        pred = row.get("pred", "").lower().strip()
        gold = row.get("answer", "").lower().strip()
        return float(gold in pred)