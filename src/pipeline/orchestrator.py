from logging import DEBUG

import hydra
from hydra.core.hydra_config import HydraConfig
from src.utils.experiment_history import append_experiment_row
from omegaconf import DictConfig
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import open_dict

class ExperimentRunner:
    """
    Orchestrateur générique du pipeline RAG.
    Lit les stages et modules définis dans le YAML.
    Chaque module doit avoir une méthode .run().
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.root_dir = Path(hydra.utils.get_original_cwd()).resolve()

    # --------------------------------------------------------
    def run(self):
        print("\n🚀 Starting dynamic RAG pipeline")

        results = {}
        for i, stage_name in enumerate(self.cfg.stages):
            print(f"\n🔹 Stage: {stage_name}")

            if stage_name not in self.cfg.modules:
                raise ValueError(f"❌ No module config found for stage '{stage_name}'")

            module_cfg = self.cfg.modules[stage_name]

            # 🔁 Injection automatique du résultat précédent
            prev_result = results.get(self.cfg.stages[i - 1], {}) if i > 0 else {}

            for key in ["index_dir"]:  # 🔹 seules les clés qu'on veut propager
                if key in prev_result:
                    with open_dict(module_cfg):
                        module_cfg[key] = prev_result[key]

            # 🧩 Instanciation dynamique du module
            module = instantiate(module_cfg)

            if not hasattr(module, "run"):
                raise AttributeError(f"Module {module.__class__.__name__} has no .run() method")

            # 🧠 Appel universel : on passe toujours previous
            #print(results)
            result = module.run(previous=results)


            # 💾 Sauvegarde du résultat du stage
            results[stage_name] = result
            print(f"✅ Stage '{stage_name}' done.\n")

        print("\n🏁 Pipeline completed successfully.")
        return results
