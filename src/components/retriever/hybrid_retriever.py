# src/rag/retriever/hybrid.py
# --- TO DO : Add Normalization of Sparse and Dense score ---
from pathlib import Path
from src.components.retriever.base_retriever import RetrieverBase
from src.components.retriever.dense_faiss import FaissRetriever
from src.components.retriever.sparse_retriever import BM25Retriever

class HybridRetriever(RetrieverBase):
    """
    Hybrid retriever combining dense (FAISS) and sparse (BM25) retrieval scores.
    Balances both sources using a weighting factor alpha.
    """

    # Initialize the hybrid retriever with both dense and sparse components.
    # The alpha parameter controls the weighting between dense and sparse scores.
    def __init__(self, alpha=0.5, top_k=5, model_name=None, index_dir: Path = None, docs_path: Path = None):
        """
        Args:
            alpha (float): Weighting factor between dense and sparse scores (0–1).
            top_k (int): Default number of results to return.
            model_name (str): Name of the embedding model for dense retrieval.
            index_dir (Path): Directory containing FAISS index and document metadata.
            docs_path (Path): Path to JSONL file used by the BM25 retriever.
        """
        self.alpha = alpha
        self.top_k = top_k
        self.dense = FaissRetriever(index_dir= index_dir,model_name=model_name)
        self.sparse = BM25Retriever(docs_path)
        self.retrieval_type = "hybrid"

    # Retrieve top-k documents using both dense and sparse retrievers.
    # Merge results by combining normalized scores based on alpha.
    def retrieve(self, query: str, top_k: int = None):
        top_k = top_k or self.top_k
        dense_results = self.dense.retrieve(query, top_k * 2)
        sparse_results = self.sparse.retrieve(query, top_k * 2)

        # Merge dense and sparse results into a unified ranking based on weighted scores.
        all_texts = {r["text"] for r in dense_results + sparse_results}
        merged = []
        for text in all_texts:
            d_score = next((r["score"] for r in dense_results if r["text"] == text), 0)
            s_score = next((r["score"] for r in sparse_results if r["text"] == text), 0)
            score = self.alpha * d_score + (1 - self.alpha) * s_score
            merged.append({"text": text, "score": score})

        # Sort merged results by final score and return the top-k items.
        merged.sort(key=lambda r: r["score"], reverse=True)
        return merged[:top_k]

    # Return metadata describing the hybrid retriever configuration.
    def get_info(self):
        return {"type": "hybrid", "alpha": self.alpha, "num_docs": len(self.dense.docs)}
