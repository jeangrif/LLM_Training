# src/rag/retriever/hybrid.py
import numpy as np
from pathlib import Path
from src.rag.retriever.base_retriever import RetrieverBase
from src.rag.retriever.dense_faiss import FaissRetriever
from src.rag.retriever.sparse_retriever import BM25Retriever

class HybridRetriever(RetrieverBase):
    def __init__(self, alpha=0.5, top_k=5, model_name=None, index_dir: Path = None, docs_path: Path = None):
        self.alpha = alpha
        self.top_k = top_k
        self.dense = FaissRetriever(index_dir= index_dir,model_name=model_name)
        self.sparse = BM25Retriever(docs_path)
        self.retrieval_type = "hybrid"

    def retrieve(self, query: str, top_k: int = None):
        top_k = top_k or self.top_k
        dense_results = self.dense.retrieve(query, top_k * 2)
        sparse_results = self.sparse.retrieve(query, top_k * 2)

        # fusion
        all_texts = {r["text"] for r in dense_results + sparse_results}
        merged = []
        for text in all_texts:
            d_score = next((r["score"] for r in dense_results if r["text"] == text), 0)
            s_score = next((r["score"] for r in sparse_results if r["text"] == text), 0)
            score = self.alpha * d_score + (1 - self.alpha) * s_score
            merged.append({"text": text, "score": score})
        merged.sort(key=lambda r: r["score"], reverse=True)
        return merged[:top_k]

    def get_info(self):
        return {"type": "hybrid", "alpha": self.alpha, "num_docs": len(self.dense.docs)}
