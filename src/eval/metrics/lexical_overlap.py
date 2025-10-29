from src.eval.metrics.base_metric import BaseMetric

class LexicalOverlap(BaseMetric):
    def compute(self, row, group=None):
        pred, gold = row.get("pred", ""), row.get("answer", "")
        set_pred = set(pred.lower().split())
        set_gold = set(gold.lower().split())
        return len(set_pred & set_gold) / len(set_gold) if set_gold else 0.0
