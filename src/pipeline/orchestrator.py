from logging import DEBUG

import hydra
from hydra.core.hydra_config import HydraConfig
from src.utils.experiment_history import append_experiment_row
from omegaconf import DictConfig
from pathlib import Path

from hydra.utils import instantiate
from rich.console import Console
from omegaconf import open_dict
from rich.table import Table

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
        console = Console()
        console.print("\n🚀 [bold cyan]Starting dynamic RAG pipeline[/bold cyan]")

        # --- 🎨 Affichage du flow du pipeline ---
        stages = self.cfg.stages
        table = Table(title="🔁 Pipeline Flow", show_header=True, header_style="bold magenta")
        table.add_column("Order", justify="center")
        table.add_column("Stage Name", style="cyan")
        table.add_column("Module Target", style="green")

        for i, stage_name in enumerate(stages):
            module_cfg = self.cfg.modules.get(stage_name, {})
            target = module_cfg.get("_target_", "❓ Missing target")
            table.add_row(str(i + 1), stage_name, target)

        console.print(table)

        # --- 🧠 Exécution réelle ---
        results = {}
        for i, stage_name in enumerate(stages):
            console.print(f"\n🔹 [bold yellow]Stage:[/bold yellow] {stage_name}")

            if stage_name not in self.cfg.modules:
                raise ValueError(f"❌ No module config found for stage '{stage_name}'")

            module_cfg = self.cfg.modules[stage_name]

            # 🔁 Injection du résultat précédent
            prev_result = results.get(stages[i - 1], {}) if i > 0 else {}

            for key in ["index_dir"]:  # seules les clés qu'on veut propager
                if key in prev_result:
                    with open_dict(module_cfg):
                        module_cfg[key] = prev_result[key]

            # 🧩 Instanciation dynamique
            module = instantiate(module_cfg)

            if not hasattr(module, "run"):
                raise AttributeError(f"Module {module.__class__.__name__} has no .run() method")

            # 🚀 Exécution du stage
            result = module.run(previous=results)
            results[stage_name] = result

            console.print(f"✅ [green]Stage '{stage_name}' completed successfully.[/green]\n")

        console.print("\n🏁 [bold green]Pipeline completed successfully![/bold green]")
        return results
