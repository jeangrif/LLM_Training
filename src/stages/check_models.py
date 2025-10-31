# src/rag/check_models.py
from pathlib import Path
from datetime import datetime
from huggingface_hub import snapshot_download, hf_hub_download
import json


def _local_models_dir(subdir: str) -> Path:
    # Return the path to a subdirectory inside ./models/ (embed / llm / rerank).
    # Creates the directory if it does not exist.
    root = Path("models").resolve() / subdir
    root.mkdir(parents=True, exist_ok=True)
    return root


# --------------------------------------------------------
# Embedding Models
# --------------------------------------------------------

def ensure_local_embedding_model(model_name: str, download_if_missing=True) -> Path:
    """
    Check if the embedding model exists locally under ./models/embed/.
    If missing, download it from the Hugging Face Hub.
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
# LLM Models
# --------------------------------------------------------

def ensure_local_llm(repo_id: str, filename: str, download_if_missing=True) -> Path:
    """
    Check if the GGUF LLM file exists locally under ./models/llm/.
    If missing, download it from the Hugging Face Hub.
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
# Rerank Models (optional)
# --------------------------------------------------------

def ensure_local_reranker(model_name: str, download_if_missing=True) -> Path:
    """
    Check if the reranker model (CrossEncoder) exists locally under ./models/rerank/.
    If missing, download it from the Hugging Face Hub.
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
# Stage Class
# --------------------------------------------------------
class CheckModels:
    """
    Stage 1: Verify that all required models (embedding, LLM, reranker) are available locally.
    Downloads missing models if necessary.
    """
    def __init__(self, embed_model: str, llm_repo: str, llm_filename: str, rerank_model: str = None):
        self.embed_model = embed_model
        self.llm_repo = llm_repo
        self.llm_filename = llm_filename
        self.rerank_model = rerank_model

    # Check the local availability of all models defined in the configuration.
    # Returns their resolved paths and metadata for later pipeline stages.
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

        print("✅ All llm verified and available locally.")
        return {
            "status": "ok",
            "models_dir": str(Path("llm").resolve()),
            "metadata": meta,
        }
