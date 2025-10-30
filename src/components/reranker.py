# src/rag/retriever/reranker.py
# --- TO DO : Improve Reranker class, instantiate with config parameter, change top_k thanks to config parameter --
from sentence_transformers import CrossEncoder

class ReRanker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(self, query, results, top_k=None):
        test_top_k =5
        pairs = [(query, r["text"]) for r in results]
        scores = self.model.predict(pairs)
        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        if test_top_k:
            results = results[:test_top_k]
        return results
