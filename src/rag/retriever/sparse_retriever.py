# src/rag/retriever/sparse_bm25.py
from rank_bm25 import BM25Okapi
import json
from pathlib import Path
from src.rag.retriever.base_retriever import RetrieverBase

class BM25Retriever(RetrieverBase):
    def __init__(self, docs_path: Path):
        with open(docs_path, "r", encoding="utf-8") as f:
            self.docs = [json.loads(line) for line in f]
        self.corpus = [d["text"].split() for d in self.docs]
        self.bm25 = BM25Okapi(self.corpus)
        self.retrieval_type = "sparse_bm25"

    def retrieve(self, query: str, top_k: int = 5):
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)
        top_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_ids:
            results.append({
                "text": self.docs[idx]["text"],
                "score": float(scores[idx]),
                "metadata": {k: v for k, v in self.docs[idx].items() if k != "text"}
            })
        return results

    def get_info(self):
        return {"type": self.retrieval_type, "num_docs": len(self.docs)}
