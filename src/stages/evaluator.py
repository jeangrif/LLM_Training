import importlib
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from hydra.core.hydra_config import HydraConfig
from src.utils.jsonl_helper import load_jsonl, save_jsonl
import json


class Evaluator:
    def __init__(
        self,
        results_path: Path,
        output_path: Path,
        metrics: dict,
        embedding_model=None,
        save_metrics: bool = True,
        output_filename: str = "eval_metrics.jsonl",
    ):
        self.results_path = Path(results_path)
        self.save_metrics = save_metrics
        self.output_path = Path(HydraConfig.get().run.dir) / output_filename

        # Charger le modèle SentenceTransformer (partagé entre toutes les métriques)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(embedding_model)

        # Charger dynamiquement les classes de métriques
        self.metrics = self._load_metrics(metrics)
        print(f"✅ Loaded metrics: {list(self.metrics.keys())}")

    # --------------------------------------------------------
    def _load_metrics(self, metrics_cfg: dict):
        """Charge dynamiquement les classes de métriques depuis leur chemin."""
        metrics = {}
        for name, path in metrics_cfg.items():
            module_name, class_name = path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            metric_class = getattr(module, class_name)
            metrics[name] = metric_class(model=self.model)
        return metrics

    # --------------------------------------------------------
    def run(self, previous=None, **kwargs):
        # 1) Charger les résultats
        results = None
        if previous and "run_rag" in previous:
            results = previous["run_rag"].get("results")
        elif results is None:
            if not self.results_path:
                raise ValueError("No in-memory results and no results_path provided.")
            results = load_jsonl(self.results_path)

        # 2) Grouper par orig_id
        groups = defaultdict(list)
        for r in results:
            groups[r["orig_id"]].append(r)

        # 3) Évaluer chaque groupe
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

            # robustesse sur le groupe entier
            if "robustness" in self.metrics:
                try:
                    rob = self.metrics["robustness"].compute(None, group)
                    for r in group:
                        r["robustness"] = rob
                except Exception as e:
                    print(f"⚠️ Error in robustness: {e}")

            all_results.extend(group)

        # 4) Sauvegarder les résultats
        save_jsonl(all_results, self.output_path)
        print(f"✅ Evaluation results saved to {self.output_path}")

        # 5) Moyennes globales
        metric_names = list(self.metrics.keys())
        avg = {
            m: float(np.mean([r[m] for r in all_results if r.get(m) is not None]))
            for m in metric_names
        }

        print("\n📊 Average Scores:")
        for k, v in avg.items():
            print(f"  {k:<15}: {v:.4f}")

        # 6) Mise à jour du résumé expérimental
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
