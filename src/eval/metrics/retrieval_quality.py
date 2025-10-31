from collections import defaultdict
from src.utils.jsonl_helper import load_jsonl
import math

class Retriever_Quality:
    def __init__(self, model=None, qrels_path=None):
        self.qrels_path = qrels_path
        self.qrels = self._load_qrels(qrels_path)

    def _load_qrels(self, qrels_path):
        qrels = load_jsonl(qrels_path)
        qrels_dict = defaultdict(dict)
        for item in qrels:
            qrels_dict[item["qid"]][item["doc_id"]] = item.get("rel", 1)
        return qrels_dict

    def _precision_recall_f1(self, retrieved_docs, relevant_docs):
        intersection = len(set(retrieved_docs) & relevant_docs)
        precision = intersection / len(retrieved_docs) if retrieved_docs else 0.0
        recall = intersection / len(relevant_docs) if relevant_docs else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return precision, recall, f1

    def _ndcg(self, retrieved_docs, relevant_docs):
        """Compute normalized DCG (discounted cumulative gain)."""
        gains = [1 if doc in relevant_docs else 0 for doc in retrieved_docs]
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
        ideal_gains = sorted(gains, reverse=True)
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
        return dcg / idcg if idcg > 0 else 0.0

    def compute(self, row, group=None):
        qid = row["orig_id"]
        retrieved_docs = row.get("doc_ids", [])
        relevant_docs = set(self.qrels.get(qid, {}).keys())

        if not relevant_docs:
            return None  # no ground truth relevance

        precision, recall, f1 = self._precision_recall_f1(retrieved_docs, relevant_docs)
        ndcg = self._ndcg(retrieved_docs, relevant_docs)

        # top-k variant
        top_k = min(5, len(retrieved_docs))
        retrieved_topk = retrieved_docs[:top_k]
        precision_at_k, recall_at_k, _ = self._precision_recall_f1(retrieved_topk, relevant_docs)

        return {
            "qid": qid,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "precision@5": precision_at_k,
            "recall@5": recall_at_k,
            "nDCG": ndcg,
            "n_relevant": len(relevant_docs),
            "n_retrieved": len(retrieved_docs)
        }
