# -*- coding: utf-8 -*-
import os
from pathlib import Path
from platformdirs import user_cache_dir
from huggingface_hub import hf_hub_download
import os, logging
from huggingface_hub import logging as hf_logging
APP_NAME = "LLMTraining"

def _models_dir() -> Path:
    root = Path(user_cache_dir(APP_NAME)).resolve() / "llm"
    root.mkdir(parents=True, exist_ok=True)
    return root

def ensure_gguf_file(repo_id: str, filename: str) -> Path:
    """
    Offline-first: si le .gguf est déjà présent dans le cache app, on le réutilise.
    Sinon on télécharge UNIQUEMENT ce fichier (reprise incluse).
    Retourne le chemin complet vers le .gguf.
    """
    cache_dir = _models_dir()
    local = cache_dir / repo_id.replace("/", "__") / filename  # chemin déterministe
    if local.exists() and local.stat().st_size > 0:
        return local

    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        force_download=False,
    )

    return Path(path)
