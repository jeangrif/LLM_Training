import json
from pathlib import Path
from omegaconf import DictConfig

# Load a JSONL file into memory as a list of Python dictionaries.
def load_jsonl(path):
    if path is None:
        raise ValueError("❌ load_jsonl() called with path=None")

    # ✅ Compatibilité Hydra + Path + string
    if isinstance(path, DictConfig):
        path = str(path)
    elif isinstance(path, Path):
        path = str(path)
    elif not isinstance(path, str):
        raise TypeError(f"❌ Unsupported path type: {type(path)}")

    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# Save a list of Python dictionaries to a JSONL file (one record per line).
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")