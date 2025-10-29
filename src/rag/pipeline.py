# src/rag/pipeline.py

from pathlib import Path
from src.rag.retriever.dense_faiss import FaissRetriever
from src.rag.retriever.sparse_retriever import BM25Retriever
from src.rag.retriever.hybrid_retriever import HybridRetriever
from src.rag.reranker import ReRanker
from src.rag.generator import RagGenerator
from src.eval.metrics.latency import LatencyMeter
from hydra.core.hydra_config import HydraConfig
import json

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
        latency_cfg = None
    ):
        self.top_k = top_k
        self.retrieval_type = retrieval_type
        self.use_rerank = use_rerank
        self.alpha = alpha
        self.docs_path = Path(index_dir) / "docs.jsonl"
        self.model_meta = model_meta
        self.latency_cfg = latency_cfg or {}
        self.latency_meter = LatencyMeter() if self.latency_cfg.get("enabled", True) else None

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
        """Pipeline complet pour une seule question avec mesure de latence"""

        # --- Retrieval ---
        if self.latency_meter:
            self.latency_meter.start("retrieval")
        results = self.retriever.retrieve(query, top_k=self.top_k)
        if self.latency_meter:
            self.latency_meter.stop("retrieval")

        # --- Rerank (si activé) ---
        if self.reranker:
            if self.latency_meter:
                self.latency_meter.start("rerank")
            results = self.reranker.rerank(query, results, top_k=self.top_k)
            if self.latency_meter:
                self.latency_meter.stop("rerank")

        # --- Generation ---
        contexts = [r["text"] for r in results]
        if self.latency_meter:
            self.latency_meter.start("generation")
        answer = self.generator.generate(query, contexts)
        if self.latency_meter:
            self.latency_meter.stop("generation")

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

    def summarize_latency(self, output_dir: Path = None):
        """Sauvegarde le résumé de latence dans le dossier de l'expérimentation Hydra."""
        if not self.latency_meter:
            return

        # Récupération du dossier Hydra courant si non spécifié
        if output_dir is None:
            output_dir = Path(HydraConfig.get().run.dir)

        summary = self.latency_meter.summary()
        output_path = Path(output_dir) / "summarize_experiment.json"

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"✅ Latency summary saved to {output_path}")


