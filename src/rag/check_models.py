# src/rag/check_models.py
from pathlib import Path
from datetime import datetime
from huggingface_hub import snapshot_download, hf_hub_download
import json


def _local_models_dir(subdir: str) -> Path:
    """Retourne un sous-dossier dans ./models/ (embed / llm / rerank)."""
    root = Path("models").resolve() / subdir
    root.mkdir(parents=True, exist_ok=True)
    return root


# --------------------------------------------------------
# 🔹 EMBEDDING MODELS
# --------------------------------------------------------
def ensure_local_embedding_model(model_name: str, download_if_missing=True) -> Path:
    """
    Vérifie ou télécharge un modèle d'embedding dans ./models/embed/.
    """
    safe_name = model_name.replace("/", "__")
    local_dir = _local_models_dir("embed") / safe_name

    if (local_dir / "config.json").exists():
        return local_dir

    if not download_if_missing:
        print(f"⚠️ Embedding model not found locally: {local_dir}")
        return None

    print(f"⬇️ Downloading embedding model {model_name} → {local_dir}")
    snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)
    return local_dir


# --------------------------------------------------------
# 🔹 LLM MODELS
# --------------------------------------------------------
def ensure_local_llm(repo_id: str, filename: str, download_if_missing=True) -> Path:
    """
    Vérifie ou télécharge un modèle GGUF dans ./models/llm/.
    """
    safe_name = repo_id.replace("/", "__")
    local_dir = _local_models_dir("llm") / safe_name
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / filename

    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    if not download_if_missing:
        print(f"⚠️ LLM file missing locally: {local_path}")
        return None

    print(f"⬇️ Downloading {repo_id}/{filename} → {local_path}")
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(local_dir))
    return local_path


# --------------------------------------------------------
# 🔹 RERANK MODELS (optionnel)
# --------------------------------------------------------
def ensure_local_reranker(model_name: str, download_if_missing=True) -> Path:
    """
    Vérifie ou télécharge un modèle de re-ranking (CrossEncoder).
    """
    safe_name = model_name.replace("/", "__")
    local_dir = _local_models_dir("rerank") / safe_name

    if (local_dir / "config.json").exists():
        return local_dir

    if not download_if_missing:
        print(f"⚠️ Rerank model not found locally: {local_dir}")
        return None

    print(f"⬇️ Downloading rerank model {model_name} → {local_dir}")
    snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)
    return local_dir


# --------------------------------------------------------
# 🔹 STAGE CLASS
# --------------------------------------------------------
class CheckModels:
    """Stage 1 : vérifie la présence locale des modèles (embed / llm / rerank)."""

    def __init__(self, embed_model: str, llm_repo: str, llm_filename: str, rerank_model: str = None):
        self.embed_model = embed_model
        self.llm_repo = llm_repo
        self.llm_filename = llm_filename
        self.rerank_model = rerank_model

    def run(self, **kwargs):
        print("🔍 Checking local model availability...")

        embed_path = ensure_local_embedding_model(self.embed_model)
        llm_path = ensure_local_llm(self.llm_repo, self.llm_filename)
        rerank_path = ensure_local_reranker(self.rerank_model) if self.rerank_model else None

        meta = {
            "embedding_model": self.embed_model,
            "embedding_path": str(embed_path),
            "llm_repo": self.llm_repo,
            "llm_filename": self.llm_filename,
            "llm_path": str(llm_path),
            "rerank_model": self.rerank_model,
            "rerank_path": str(rerank_path) if rerank_path else None,
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }

        print("✅ All models verified and available locally.")
        return {
            "status": "ok",
            "models_dir": str(Path("models").resolve()),
            "metadata": meta,
        }
