import shutil
from pathlib import Path
from platformdirs import user_cache_dir

def cache_llm_if_needed(local_file: Path, repo_id: str) -> Path:
    """
    Copie le modèle depuis llm/llm/... vers le cache global (~/.cache/LLMTraining)
    uniquement s'il n'est pas déjà présent.
    """
    dst = Path(user_cache_dir("LLMTraining")) / "llm" / "llm" / repo_id.replace("/", "__") / local_file.name
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not dst.exists():
        print(f"📦 Caching LLM model {local_file.name} → {dst}")
        shutil.copy2(local_file, dst)

    return dst
