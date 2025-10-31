# src/rag/rag_runner.py

from pathlib import Path
from src.rag.engine import RagPipeline
from src.utils.jsonl_helper import load_jsonl, save_jsonl  # si tu as ces helpers
from tqdm import tqdm
from src.utils.display import display_rag_pipeline_config
from hydra.core.hydra_config import HydraConfig
import random
class RagRunner:
    """
    Execute the full RAG (Retrieval-Augmented Generation) process.
    Handles data loading, question sampling, pipeline execution, and result saving.
    """

    # Initialize the RAG runner with input/output configuration and retrieval parameters.
    # Stores metadata, paths, and model configurations required for execution.
    def __init__(
        self,
        input_path: str,
        output_path: str,
        retrieval_type: str = "dense",
        use_rerank: bool = False,
        top_k: int = 3,
        alpha: float = 0.5,
        top_k_rerank: int =5,
        max_questions: int = None,
        index_dir: str = None,
        embedding_model = None,
        save_results: bool = False,
        output_filename: str = "results_rag.jsonl",
        model_cfg=None,
        latency_cfg=None,
        do_generation: bool = True,
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
        self.do_generation = do_generation
        self.top_k_rerank = top_k_rerank

    # ------------------------------------------------------------
    def run(self, previous=None, **kwargs):
        """
        Run the complete RAG workflow.

        Loads input data, initializes the RAG pipeline, processes each query,
        and optionally saves the generated results.

        Args:
            previous: Dictionary containing outputs from previous pipeline stages.

        Returns:
            Dictionary with the generated results and optional output file path.
        """
        print(f"🧠 Running RAG on file: {self.input_path.name}")

        # Display the current RAG configuration (retrieval mode, reranking, parameters).
        display_rag_pipeline_config(
            retrieval_type=self.retrieval_type,
            use_rerank=self.use_rerank,
            top_k=self.top_k,
            alpha=self.alpha,
            index_dir=self.index_dir,
            embedding_model=self.embedding_model,
            do_generation=self.do_generation,
            top_k_rerank = self.top_k_rerank
        )
        data = load_jsonl(self.input_path)

        # Optionally limit the number of processed questions for faster experimentation or evaluation.
        if self.max_questions:
            data = random.sample(data, min(self.max_questions, len(data)))

        # Initialize the core RAG pipeline that handles retrieval and generation logic.
        pipeline = RagPipeline(
            top_k=self.top_k,
            retrieval_type=self.retrieval_type,
            use_rerank=self.use_rerank,
            alpha=self.alpha,
            index_dir=self.index_dir,
            embedding_model = self.embedding_model,
            model_meta=previous.get("check_models", {}).get("metadata", None),
            model_cfg=self.model_cfg,
            latency_cfg = self.latency_cfg,
            do_generation=self.do_generation,
            top_k_rerank = self.top_k_rerank
        )

        results = []
        # Process each question through the RAG pipeline and collect generated answers.
        for item in tqdm(data, desc="Processing questions"):
            q = item["question"]
            r = pipeline.run(q)
            r["orig_id"] = item.get("orig_id", None)
            r["degree"] = item.get("degree", None)
            r["answer"] = item.get("answer", "")
            results.append(r)

        # Gracefully release model resources and GPU memory if supported by the pipeline.
        if hasattr(pipeline, "close"):
            pipeline.close()
        results_path = None

        # Optionally display latency statistics if the pipeline provides them.
        if hasattr(pipeline, "summarize_latency"):
            pipeline.summarize_latency()

        # Save generated results to a JSONL file if saving is enabled.
        if self.save_results:
            results_path = self.run_dir / self.output_filename
            save_jsonl(results, results_path)
            print(f"✅ RAG saved → {results_path}")

        return {
            "results": results,
            "results_path": str(results_path) if results_path else None,
        }

