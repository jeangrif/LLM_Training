import json
from pathlib import Path

# Load a JSONL file into memory as a list of Python dictionaries.
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# Save a list of Python dictionaries to a JSONL file (one record per line).
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")