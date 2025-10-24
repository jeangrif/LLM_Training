import inspect
import importlib
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from hydra.core.hydra_config import HydraConfig
from src.utils.jsonl_helper import load_jsonl, save_jsonl


class Evaluator:
    def __init__(self, results_path: Path, output_path: Path, metrics: dict, embedding_model=None,save_metrics: bool = True,output_filename: str = "eval_metrics.jsonl"):
        self.results_path = Path(results_path)
        self.output_path = Path(output_path)
        self.metrics = self._load_metrics(metrics)
        self.save_metrics = save_metrics
        self.output_path = Path(HydraConfig.get().run.dir) / output_filename

        # Charger ton modèle une seule fois pour les métriques qui en ont besoin
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(embedding_model)

        print(f"✅ Loaded metrics: {list(self.metrics.keys())}")

    # --------------------------------------------------------
    def _load_metrics(self, metrics_cfg: dict):
        """Charge dynamiquement les fonctions de métriques depuis leur chemin."""
        metrics = {}
        for name, path in metrics_cfg.items():
            module_name, func_name = path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            metrics[name] = getattr(module, func_name)
        return metrics

    # --------------------------------------------------------
    def _safe_call(self, func, **kwargs):
        """Appelle une métrique même si elle ne prend pas tous les arguments."""
        sig = inspect.signature(func)
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return func(**valid_kwargs)

    # --------------------------------------------------------
    def run(self, previous=None, **kwargs):
        results = None
        if previous and "run_rag" in previous:
            results = previous["run_rag"].get("results")

        # 2) sinon on lit le fichier si fourni
        elif results is None:
            if not self.results_path:
                raise ValueError("No in-memory results and no results_path provided.")
            results = load_jsonl(self.results_path)

        # group by orig_id
        groups = defaultdict(list)
        for r in results:
            groups[r["orig_id"]].append(r)

        all_results = []
        for gid, group in tqdm(groups.items(), desc="Evaluating", leave=False):
            for row in group:
                pred, gold = row.get("pred", ""), row.get("answer", "")
                contexts = row.get("contexts", [])

                for name, func in self.metrics.items():
                    if name == "robustness":
                        continue  # géré après le groupement
                    try:
                        row[name] = self._safe_call(
                            func,
                            pred=pred,
                            gold=gold,
                            contexts=contexts,
                            group=group,
                            model=self.model,
                        )
                    except Exception as e:
                        row[name] = None
                        print(f"⚠️ Error in {name}: {e}")

            # robustesse sur le groupe entier
            if "robustness" in self.metrics:
                try:
                    rob = self._safe_call(self.metrics["robustness"], group=group, model=self.model)
                    for r in group:
                        r["robustness"] = rob
                except Exception as e:
                    print(f"⚠️ Error in robustness: {e}")

            all_results.extend(group)

        save_jsonl(all_results, self.output_path)
        print(f"✅ Evaluation results saved to {self.output_path}")

        # Calcul des moyennes
        metric_names = list(self.metrics.keys())
        avg = {
            m: np.mean([r[m] for r in all_results if r.get(m) is not None])
            for m in metric_names
        }
        print("\n📊 Average Scores:")
        for k, v in avg.items():
            print(f"  {k:<15}: {v:.4f}")
        return avg
