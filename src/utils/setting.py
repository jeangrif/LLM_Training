from pathlib import Path
from omegaconf import DictConfig


class RagSettings:
    """
    Hydra-native settings handler.
    Used to initialize and verify directory structures before running the pipeline.
    """
    def __init__(self, cfg: DictConfig):
        """
        Initialize settings from a Hydra configuration.

        Args:
            cfg: Hydra DictConfig object corresponding to embed.settings.yaml.
        """
        self.cfg = cfg
        self.raw_dir = Path(cfg.get("raw_dir", "data/raw"))
        self.index_dir = Path(cfg.get("index_dir", "data/index"))

    def ensure_dirs(self):
        """
        Create required directories (raw and index) if they do not already exist.
        """
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

