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
    Generic orchestrator for the RAG pipeline.
    Loads the pipeline stages from the Hydra YAML configuration and executes them sequentially.
    Each module must implement a `.run()` method returning its output.
    """

    # Initialize the ExperimentRunner with the Hydra configuration.
    # Stores the root directory and configuration for later access.
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.root_dir = Path(hydra.utils.get_original_cwd()).resolve()

    # --------------------------------------------------------
    # Execute the full pipeline based on the defined stages.
    # Displays the pipeline flow, dynamically instantiates each module,
    # and propagates outputs between stages when needed.
    def run(self):
        console = Console()
        console.print("\n🚀 [bold cyan]Starting dynamic RAG pipeline[/bold cyan]")

        # Display the ordered list of pipeline stages with their target modules.
        # Provides a visual summary of the configured execution flow.
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

        # Sequentially execute each stage of the pipeline.
        # Each stage receives optional results from the previous stage when required.
        results = {}
        for i, stage_name in enumerate(stages):
            console.print(f"\n🔹 [bold yellow]Stage:[/bold yellow] {stage_name}")

            if stage_name not in self.cfg.modules:
                raise ValueError(f"❌ No module config found for stage '{stage_name}'")

            module_cfg = self.cfg.modules[stage_name]

            # Inject relevant outputs (e.g., paths, indexes) from the previous stage
            # into the current module configuration to maintain state continuity.
            prev_result = {}
            for past_stage in stages[:i]:
                prev_result.update(results.get(past_stage, {}))

            keys_to_propagate = module_cfg.get("propagate_keys", [])
            for key in keys_to_propagate:
                if key in prev_result:
                    with open_dict(module_cfg):
                        module_cfg[key] = prev_result[key]
            with open_dict(module_cfg):
                module_cfg.pop("propagate_keys", None)


        
            # Dynamically instantiate the module defined in the YAML config using Hydra.
            module = instantiate(module_cfg)

            if not hasattr(module, "run"):
                raise AttributeError(f"Module {module.__class__.__name__} has no .run() method")


            # Execute the current module's `.run()` method and store its results.
            result = module.run(previous=results)
            results[stage_name] = result


            console.print(f"✅ [green]Stage '{stage_name}' completed successfully.[/green]\n")

        console.print("\n🏁 [bold green]Pipeline completed successfully![/bold green]")
        return results
