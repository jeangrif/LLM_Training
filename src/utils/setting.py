from pathlib import Path
from omegaconf import DictConfig


class RagSettings:
    """
    Version Hydra-native du chargeur de settings.yaml.
    Sert uniquement à préparer les répertoires avant un run.
    """

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: DictConfig Hydra correspondant à ${embed.settings.yaml}
        """
        self.cfg = cfg
        self.raw_dir = Path(cfg.get("raw_dir", "data/raw"))
        self.index_dir = Path(cfg.get("index_dir", "data/index"))

    def ensure_dirs(self):
        """Crée les répertoires nécessaires si absents"""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        #print(f"📁 Ensured dirs: {self.raw_dir} / {self.index_dir}")
