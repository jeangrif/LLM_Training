# src/rag/engine.py
# --- TO DO : Improve retriever handling, should be more structured not if/else cond ---
from pathlib import Path
from src.components.retriever.dense_faiss import FaissRetriever
from src.components.retriever.sparse_retriever import BM25Retriever
from src.components.retriever.hybrid_retriever import HybridRetriever
from src.components.reranker import ReRanker
from src.components.generator.generator import RagGenerator
from src.eval.performance.latency import LatencyMeter
from hydra.core.hydra_config import HydraConfig
from src.utils.scoring import attach_unified_scores
import json

class RagPipeline:
    """
    Complete RAG pipeline: retrieval → (optional reranking) → generation.
    Supports both Hydra-integrated and standalone usage.
    """

    # Initialize the RAG pipeline components based on configuration.
    # Handles retriever selection, optional reranker, generator setup, and latency tracking.
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
        do_generation: bool = True,
        latency_cfg = None,
        top_k_rerank : int=5,
        stateful: bool = False,
        guardrails_cfg=None
    ):
        self.top_k = top_k
        self.retrieval_type = retrieval_type
        self.use_rerank = use_rerank
        self.alpha = alpha
        self.docs_path = Path(index_dir) / "docs.jsonl"
        self.model_meta = model_meta
        self.latency_cfg = latency_cfg or {}
        self.latency_meter = LatencyMeter() if self.latency_cfg.get("enabled", True) else None
        self.do_generation = do_generation
        self.top_k_rerank = top_k_rerank
        self.stateful = stateful
        self.model_cfg = model_cfg
        self.guardrails_cfg = guardrails_cfg

        # Initialize the retrieval component based on the selected retrieval strategy:
        # - dense: FAISS vector-based retrieval
        # - sparse: BM25 lexical retrieval
        # - hybrid: combined dense + sparse guardrails with weighting factor alpha
        if retrieval_type == "dense":
            self.retriever = FaissRetriever(index_dir=index_dir, parquet_path=parquet_path, model_name=embedding_model)
        elif retrieval_type == "sparse":
            self.retriever = BM25Retriever(index_dir=index_dir)
        elif retrieval_type == "hybrid":
            self.retriever = HybridRetriever(alpha=alpha, top_k=top_k, model_name=embedding_model,index_dir=index_dir, docs_path =self.docs_path )
        else:
            raise ValueError(f"Unknown retrieval type: {retrieval_type}")

        # Initialize the reranker if enabled; otherwise skip.
        # The reranker refines retrieval results before generation.
        self.reranker = ReRanker() if use_rerank else None
        self.generator = None
        if self.do_generation:
            self.generator = RagGenerator(self.model_meta, model_cfg=model_cfg, stateful=stateful)

    def run(self, query: str) -> dict:
        """
        Execute the full RAG process for a single query.

        Steps:
            1. Retrieve candidate contexts.
            2. Optionally rerank results for improved relevance.
            3. Generate the final answer using the retrieved contexts.

        Returns:
            dict: Contains the input query, generated answer, and retrieved contexts.
        """
        # Retrieve top-k relevant contexts from the index using the selected retriever.
        # Measure latency for the retrieval step if enabled.
        if self.latency_meter:
            self.latency_meter.start("retrieval")
        results = self.retriever.retrieve(query, top_k=self.top_k)
        if self.latency_meter:
            self.latency_meter.stop("retrieval")

        # Optionally rerank retrieved contexts to improve semantic relevance.
        # Measure latency for the reranking step if enabled.
        if self.reranker:
            if self.latency_meter:
                self.latency_meter.start("rerank")
            results = self.reranker.rerank(query, results, top_k=self.top_k_rerank)
            if self.latency_meter:
                self.latency_meter.stop("rerank")

        # Generate the final answer using the query and the retrieved (and possibly reranked) contexts.
        # Measure latency for the generation step if enabled.
        contexts = [r["text"] for r in results]
        doc_ids = [r["doc_id"] for r in results]
        scores = attach_unified_scores(results, self.retrieval_type, self.alpha, guardrails_cfg=self.guardrails_cfg)
        min_score = float(self.model_cfg.get("min_score", 0.0)) if hasattr(self, "model_cfg") else 0.0

        triplets = [(d, c, s) for d, c, s in zip(doc_ids, contexts, scores) if s >= min_score]

        triplets.sort(key=lambda t: t[2], reverse=True)

        # 4) Construire les docs attendus par PromptBuilder.build(query, docs)
        docs_for_gen = [
            {"doc_id": d, "text": c, "score": float(s)}
            for d, c, s in triplets
        ]



        answer = None
        if self.do_generation:
            if self.latency_meter:
                self.latency_meter.start("generation")
            answer = self.generator.generate(query, docs_for_gen)
            if self.latency_meter:
                self.latency_meter.stop("generation")
            # Reset generator state if supported (useful for session-based models).

            if not self.stateful and hasattr(self.generator, "reset"):
                self.generator.reset()

        return {"query": query, "pred": answer, "contexts": [d["text"] for d in docs_for_gen],
                    "doc_ids": [d["doc_id"] for d in docs_for_gen], }
    def close(self):
        """
        Cleanly release model resources (GPU memory, file handles, etc.) for generator and retriever.
        """
        if hasattr(self.generator, "close"):
            self.generator.close()
        if hasattr(self.retriever, "close"):
            try:
                self.retriever.close()
            except Exception:
                pass

    def reset_context(self):
        if hasattr(self.generator, "reset"):
            self.generator.reset()

    def summarize_latency(self, output_dir: Path = None):
        """
        Save a latency summary report in the current Hydra experiment directory.
        """
        if not self.latency_meter:
            return

        # Use Hydra's current run directory if no output directory is explicitly provided.
        if output_dir is None:
            output_dir = Path(HydraConfig.get().run.dir)

        summary = self.latency_meter.summary()
        output_path = Path(output_dir) / "summarize_experiment.json"

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Confirm that the latency summary file has been successfully written.
        print(f"✅ Latency summary saved to {output_path}")


