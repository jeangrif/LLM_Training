import hydra
from omegaconf import DictConfig
from src.pipeline.orchestrator import ExperimentRunner

@hydra.main(config_path="../configs", config_name="pipeline.yaml", version_base=None)
def main(cfg: DictConfig):
    runner = ExperimentRunner(cfg)
    runner.run()

if __name__ == "__main__":
    main()