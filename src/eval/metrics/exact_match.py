from src.eval.metrics.base_metric import BaseMetric

class ExactMatch(BaseMetric):
    def compute(self, row, group=None):
        pred = row.get("pred", "").lower().strip()
        gold = row.get("answer", "").lower().strip()
        return float(gold in pred)