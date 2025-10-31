# src/rag/retriever/sparse_bm25.py
from rank_bm25 import BM25Okapi
import json
from pathlib import Path
from src.components.retriever.base_retriever import RetrieverBase

class BM25Retriever(RetrieverBase):
    """
    Sparse retriever using BM25 over a preloaded text corpus.
    Ranks documents based on lexical term overlap and inverse document frequency.
    """

    # Initialize the BM25 retriever from a JSONL document file.
    # Builds the BM25 index from tokenized document texts.
    def __init__(self, docs_path: Path):
        """
        Args:
            docs_path (Path): Path to the JSONL file containing documents with a 'text' field.
        """
        with open(docs_path, "r", encoding="utf-8") as f:
            self.docs = [json.loads(line) for line in f]
        self.corpus = [d["text"].split() for d in self.docs]
        self.bm25 = BM25Okapi(self.corpus)
        self.retrieval_type = "sparse_bm25"

    # Retrieve the top-k most relevant documents for a query using BM25 scoring.
    def retrieve(self, query: str, top_k: int = 5):
        # Tokenize the query for BM25 matching (simple whitespace split).
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)
        # Select top-k document indices with the highest BM25 scores.
        top_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_ids:
            results.append({
                "text": self.docs[idx]["text"],
                "score": float(scores[idx]),
                "metadata": {k: v for k, v in self.docs[idx].items() if k != "text"}
            })
        return results

    # Return metadata about the retriever, including type and number of indexed documents.
    def get_info(self):
        return {"type": self.retrieval_type, "num_docs": len(self.docs)}
