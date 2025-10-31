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
    def __init__(
            self,
            alpha: float = 0.5,
            top_k: int = 5,
            model_name: str | None = None,
            index_dir: Path | None = None,
            docs_path: Path | None = None,
            fusion: str = "rrf",  # "rrf" | "minmax"
            k_dense: int | None = None,
            k_sparse: int | None = None,
            rrf_k: int = 60  # RRF constant; typical 10–60
    ):
        """
        Args:
            alpha (float): Weight between dense and sparse scores (used for minmax fusion).
            top_k (int): Number of final results to return.
            model_name (str): Embedding model for dense retrieval.
            index_dir (Path): Directory with FAISS index and docs.
            docs_path (Path): JSONL for BM25 corpus.
            fusion (str): Fusion strategy: "rrf" (rank-based) or "minmax" (per-query min–max).
            k_dense (int): Candidate pool size for dense retriever (defaults to 3*top_k).
            k_sparse (int): Candidate pool size for sparse retriever (defaults to 3*top_k).
            rrf_k (int): RRF damping constant (higher → smaller rank differences).
        """
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
        self.alpha = alpha
        self.top_k = top_k
        self.fusion = fusion
        self.k_dense = k_dense or (3 * top_k)
        self.k_sparse = k_sparse or (3 * top_k)
        self.rrf_k = rrf_k

        self.dense = FaissRetriever(index_dir=index_dir, model_name=model_name)
        self.sparse = BM25Retriever(docs_path)
        self.retrieval_type = "hybrid"

    # Normalize a list of scores to [0,1] per query; stable when all equal.
    def _minmax(self, scores: list[float]) -> list[float]:
        lo, hi = min(scores), max(scores)
        if hi == lo:
            return [0.0] * len(scores)
        span = hi - lo
        return [(s - lo) / span for s in scores]

    # Prefer an explicit document id to avoid merging different docs with same text.
    def _doc_id(self, rec: dict) -> str:
        meta = rec.get("metadata", {})
        sid = meta.get("source_id")
        cid = meta.get("chunk_id")
        if sid is not None and cid is not None:
            return f"{sid}:{cid}"
        # Fallback (less robust): hash of text
        return f"txt:{hash(rec['text'])}"

    # Retrieve top-k documents using both dense and sparse retrievers.
    # Fuse rankings either by RRF (scale-free) or by per-query min–max + alpha weighting.

    def retrieve(self, query: str, top_k: int | None = None):
        k = top_k or self.top_k
        d = self.dense.retrieve(query, self.k_dense)
        s = self.sparse.retrieve(query, self.k_sparse)

        # Build maps keyed by stable doc id
        d_map = {self._doc_id(r): r for r in d}
        s_map = {self._doc_id(r): r for r in s}
        ids = set(d_map) | set(s_map)

        results = []
        if self.fusion == "rrf":
            # Rank positions (1-based). Missing → very low contribution.
            d_ranks = {i: rank + 1 for rank, i in enumerate([self._doc_id(r) for r in d])}
            s_ranks = {i: rank + 1 for rank, i in enumerate([self._doc_id(r) for r in s])}
            for i in ids:
                rrf = 0.0
                if i in d_ranks:
                    rrf += 1.0 / (self.rrf_k + d_ranks[i])
                if i in s_ranks:
                    rrf += 1.0 / (self.rrf_k + s_ranks[i])
                base = d_map.get(i) or s_map.get(i)
                results.append({
                    "doc_id": i,
                    "text": base["text"],
                    "fused_score": rrf,
                    "dense_score": float(d_map.get(i, {}).get("score", 0.0)),
                    "sparse_score": float(s_map.get(i, {}).get("score", 0.0)),
                    "metadata": base.get("metadata", {})
                })
        else:  # "minmax"
            # Normalize scores to [0,1] separately, then alpha-weighted sum
            d_ids = [self._doc_id(r) for r in d]
            s_ids = [self._doc_id(r) for r in s]
            d_norm = dict(zip(d_ids, self._minmax([r["score"] for r in d]))) if d else {}
            s_norm = dict(zip(s_ids, self._minmax([r["score"] for r in s]))) if s else {}
            for i in ids:
                ds = d_norm.get(i, 0.0)
                ss = s_norm.get(i, 0.0)
                fused = self.alpha * ds + (1 - self.alpha) * ss
                base = d_map.get(i) or s_map.get(i)
                results.append({
                    "text": base["text"],
                    "fused_score": fused,
                    "dense_score": float(d_map.get(i, {}).get("score", 0.0)),
                    "sparse_score": float(s_map.get(i, {}).get("score", 0.0)),
                    "metadata": base.get("metadata", {})
                })

        # Deterministic sort, then truncate
        results.sort(key=lambda r: (r["fused_score"], r["dense_score"], r["sparse_score"], r["text"]), reverse=True)
        return results[:k]

    # Return metadata describing the hybrid retriever configuration.
    def get_info(self):
        return {
            "type": "hybrid",
            "alpha": self.alpha,
            "fusion": self.fusion,
            "k_dense": self.k_dense,
            "k_sparse": self.k_sparse,
            "num_docs": len(self.dense.docs)
        }
