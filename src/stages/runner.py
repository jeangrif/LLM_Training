# src/rag/runner.py

from pathlib import Path
from src.rag.pipeline import RagPipeline
from src.utils.jsonl_helper import load_jsonl, save_jsonl  # si tu as ces helpers
from tqdm import tqdm
from src.utils.display import display_rag_pipeline_config
from hydra.core.hydra_config import HydraConfig
import random
class RagRunner:
    def __init__(
        self,
        input_path: str,
        output_path: str,
        retrieval_type: str = "dense",
        use_rerank: bool = False,
        top_k: int = 3,
        alpha: float = 0.5,
        max_questions: int = None,
        index_dir: str = None,
        embedding_model = None,
        save_results: bool = False,
        output_filename: str = "results_rag.jsonl",
        model_cfg=None,
        latency_cfg=None,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.retrieval_type = retrieval_type
        self.use_rerank = use_rerank
        self.top_k = top_k
        self.alpha = alpha
        self.max_questions = max_questions
        self.index_dir = Path(index_dir) if index_dir else None
        self.embedding_model = embedding_model
        self.save_results = save_results
        self.output_filename = output_filename
        self.run_dir = Path(HydraConfig.get().run.dir)
        self.model_cfg = model_cfg
        self.latency_cfg = latency_cfg

    # ------------------------------------------------------------
    def run(self, previous=None, **kwargs):
        """Exécute la boucle complète de génération RAG"""
        print(f"🧠 Running RAG on file: {self.input_path.name}")

        display_rag_pipeline_config(
            retrieval_type=self.retrieval_type,
            use_rerank=self.use_rerank,
            top_k=self.top_k,
            alpha=self.alpha,
            index_dir=self.index_dir,
            embedding_model=self.embedding_model,
        )
        data = load_jsonl(self.input_path)
        random.seed(42)
        if self.max_questions:
            data = random.sample(data, min(self.max_questions, len(data)))


        pipeline = RagPipeline(
            top_k=self.top_k,
            retrieval_type=self.retrieval_type,
            use_rerank=self.use_rerank,
            alpha=self.alpha,
            index_dir=self.index_dir,
            embedding_model = self.embedding_model,
            model_meta=previous.get("check_models", {}).get("metadata", None),
            model_cfg=self.model_cfg,
            latency_cfg = self.latency_cfg
        )

        results = []
        for item in tqdm(data, desc="Processing questions"):
            q = item["question"]
            r = pipeline.run(q)
            r["orig_id"] = item.get("orig_id", None)
            r["degree"] = item.get("degree", None)
            r["answer"] = item.get("answer", "")
            results.append(r)

        # ✅ Libération propre du modèle et du GPU
        if hasattr(pipeline, "close"):
            pipeline.close()
        results_path = None
        if hasattr(pipeline, "summarize_latency"):
            pipeline.summarize_latency()
        if self.save_results:
            results_path = self.run_dir / self.output_filename
            save_jsonl(results, results_path)
            print(f"✅ RAG saved → {results_path}")
        return {
            "results": results,
            "results_path": str(results_path) if results_path else None,
        }

