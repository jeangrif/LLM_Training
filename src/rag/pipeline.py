# src/rag/pipeline.py

from pathlib import Path
from src.rag.retriever.dense_faiss import FaissRetriever
from src.rag.retriever.sparse_retriever import BM25Retriever
from src.rag.retriever.hybrid_retriever import HybridRetriever
from src.rag.reranker import ReRanker
from src.rag.generator import RagGenerator



class RagPipeline:
    """
    Pipeline complet : retrieval → (optional rerank) → génération
    Compatible pipeline Hydra (avec index_dir) et usage autonome.
    """

    def __init__(
        self,
        top_k=3,
        retrieval_type="dense",
        use_rerank=False,
        alpha=0.5,
        index_dir: Path = None,
        parquet_path: Path = None,
        embedding_model = None,
        model_meta=None,
        model_cfg=None,
    ):
        self.top_k = top_k
        self.retrieval_type = retrieval_type
        self.use_rerank = use_rerank
        self.alpha = alpha
        self.docs_path = Path(index_dir) / "docs.jsonl"
        self.model_meta = model_meta

        # --- Choix du retriever ---
        if retrieval_type == "dense":
            self.retriever = FaissRetriever(index_dir=index_dir, parquet_path=parquet_path, model_name=embedding_model)
        elif retrieval_type == "sparse":
            self.retriever = BM25Retriever(self.docs_path)
        elif retrieval_type == "hybrid":
            self.retriever = HybridRetriever(alpha=alpha, top_k=top_k, model_name=embedding_model,index_dir=index_dir, docs_path =self.docs_path )
        else:
            raise ValueError(f"Unknown retrieval type: {retrieval_type}")

        # --- Reranker optionnel ---
        self.reranker = ReRanker() if use_rerank else None
        self.generator = RagGenerator(self.model_meta, model_cfg=model_cfg )

    def run(self, query: str) -> dict:
        """Pipeline complet pour une seule question"""
        results = self.retriever.retrieve(query, top_k=self.top_k)
        if self.reranker:
            results = self.reranker.rerank(query, results, top_k=self.top_k)
        contexts = [r["text"] for r in results]
        answer = self.generator.generate(query, contexts)
        if hasattr(self.generator, "reset"):
            self.generator.reset()
        return {"query": query, "pred": answer, "contexts": contexts}

    def close(self):
        """Libère proprement les ressources du modèle."""
        if hasattr(self.generator, "close"):
            self.generator.close()
        if hasattr(self.retriever, "close"):
            try:
                self.retriever.close()
            except Exception:
                pass
