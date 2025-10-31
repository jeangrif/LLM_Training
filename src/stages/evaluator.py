import importlib
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from hydra.core.hydra_config import HydraConfig
from src.utils.jsonl_helper import load_jsonl, save_jsonl
import json
from sentence_transformers import SentenceTransformer

class Evaluator:
    """
    Evaluate RAG results using multiple metrics.
    Loads metrics dynamically, applies them per question group, and aggregates results.
    """

    # Initialize the evaluator with results, metric definitions, and model configuration.
    # Loads the shared embedding model and dynamically instantiates each metric.
    def __init__(
        self,
        results_path: Path,
        output_path: Path,
        metrics: dict,
        embedding_model=None,
        save_metrics: bool = True,
        qrels_path: Path = None,
        output_filename: str = "eval_metrics.jsonl",
        do_generation: bool = True,
    ):
        """
        Args:
            results_path (Path): Path to the JSONL file containing RAG results.
            output_path (Path): Directory to save evaluation outputs.
            metrics (dict): Mapping of metric names to their module.class import paths.
            embedding_model (str): Model name used for embedding-based metrics.
            save_metrics (bool): Whether to save detailed metric results.
            output_filename (str): Name of the output file for per-sample metrics.
        """
        self.results_path = Path(results_path)
        self.save_metrics = save_metrics
        self.output_path = Path(HydraConfig.get().run.dir) / output_filename
        self.do_generation = do_generation

        self.qrels_path = Path(qrels_path) if qrels_path else None

        # Load the shared SentenceTransformer model (used by all metrics).
        self.model = SentenceTransformer(embedding_model)

        # Dynamically load metric classes from their configured module paths.

        self.metrics = self._load_metrics(metrics)
        print(f"✅ Loaded metrics: {list(self.metrics.keys())}")

    # Dynamically import and initialize metric classes from their module paths.
    def _load_metrics(self, metrics_cfg: dict):

        if not getattr(self, "do_generation", True):

            metrics_cfg = {
                k: v for k, v in metrics_cfg.items()
                if "retriever" in k.lower() or "retrieval" in k.lower()
            }
        metrics = {}
        for name, path in metrics_cfg.items():
            module_name, class_name = path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            metric_class = getattr(module, class_name)



            # ✅ on passe qrels_path uniquement pour retrieval_quality
            if name == "retriever_quality":

                metrics[name] = metric_class(model=self.model, qrels_path=self.qrels_path)
            else:
                metrics[name] = metric_class(model=self.model)



        return metrics

    # Main evaluation loop: compute all metrics, save results, and aggregate averages.
    def run(self, previous=None, **kwargs):
        print(f"\n[DEBUG] Evaluator starting run() with qrels_path={self.qrels_path}")

        # Load RAG results either from memory (previous stage) or from a file.
        results = None
        if previous and "run_rag" in previous:
            results = previous["run_rag"].get("results")
        elif results is None:
            if not self.results_path:
                raise ValueError("No in-memory results and no results_path provided.")
            results = load_jsonl(self.results_path)

        # Group results by original ID to evaluate related questions together.
        groups = defaultdict(list)
        for r in results:
            groups[r["orig_id"]].append(r)

        # Compute all metrics for each result and handle robustness metrics at group level.
        all_results = []
        for gid, group in tqdm(groups.items(), desc="Evaluating", leave=False):
            for row in group:
                for name, metric in self.metrics.items():
                    if name == "robustness":
                        continue  # géré après groupement
                    try:
                        row[name] = metric.compute(row, group)
                    except Exception as e:
                        row[name] = None
                        print(f"⚠️ Error in {name}: {e}")

            # Robustness metric is computed at the group level.
            if "robustness" in self.metrics:
                try:
                    rob = self.metrics["robustness"].compute(None, group)
                    for r in group:
                        r["robustness"] = rob
                except Exception as e:
                    print(f"⚠️ Error in robustness: {e}")

            all_results.extend(group)

        # Save per-sample evaluation results to JSONL.
        save_jsonl(all_results, self.output_path)
        print(f"✅ Evaluation results saved to {self.output_path}")

        # Compute average scores across all evaluated results
        metric_names = list(self.metrics.keys())
        avg = {}

        for m in metric_names:
            values = [r[m] for r in all_results if r.get(m) is not None]

            if not values:
                continue

            if isinstance(values[0], dict):
                subkeys = values[0].keys()
                avg[m] = {}
                for sub in subkeys:
                    # on ne fait la moyenne que sur les champs numériques
                    numeric_values = [v[sub] for v in values if sub in v and isinstance(v[sub], (int, float))]
                    if numeric_values:
                        avg[m][sub] = float(np.mean(numeric_values))
            else:
                avg[m] = float(np.mean(values))

        # ──────────────────────────────────────────────
        # 🧾 Display neatly formatted average results
        print("\n📊 Average Scores:")
        for m, v in avg.items():
            if isinstance(v, dict):
                print(f"\n🔹 {m.upper()}:")
                for sub, val in v.items():
                    print(f"   {sub:<15}: {val:.4f}")
            else:
                print(f"  {m:<15}: {v:.4f}")

        # ──────────────────────────────────────────────
        # 🗂 Update summarize_experiment.json
        try:
            summarize_path = Path(HydraConfig.get().run.dir) / "summarize_experiment.json"
            if summarize_path.exists():
                with open(summarize_path, "r") as f:
                    summarize_data = json.load(f)
            else:
                summarize_data = {}

            summarize_data["evaluation_metrics"] = avg

            with open(summarize_path, "w") as f:
                json.dump(summarize_data, f, indent=2)

            print(f"✅ Updated summarize_experiment.json with evaluation metrics → {summarize_path}")
        except Exception as e:
            print(f"⚠️ Could not update summarize_experiment.json: {e}")

        return avg
