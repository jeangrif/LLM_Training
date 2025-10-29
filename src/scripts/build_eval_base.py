#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: Build the base evaluation dataset (id, question, answer)
from the ingested Hugging Face dataset (Parquet file).
"""

import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# --------------------------------------------
# ⚙️ Configuration (edit here as needed)
# --------------------------------------------
CONFIG = {
    "hf_dataset_name": "squad",          # must match what you ingested
    "hf_dataset_split": "train",
    "raw_dir": Path("data/raw"),         # where Parquet files are stored
    "eval_dir": Path("data/eval"),       # output directory for evaluation files
    "eval_filename": "base_questions.jsonl",
    "eval_limit": None,                  # e.g., 200 for sampling or None for full dataset
}

# --------------------------------------------
# 🚀 Main function
# --------------------------------------------
def build_eval_base():
    """Extract (id, question, answer) triplets from the Parquet dataset."""
    raw_path = CONFIG["raw_dir"] / f"{CONFIG['hf_dataset_name']}_{CONFIG['hf_dataset_split']}.parquet"
    out_path = CONFIG["eval_dir"] / CONFIG["eval_filename"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"🔹 Loading dataset: {raw_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing Parquet file: {raw_path}")

    df = pd.read_parquet(raw_path)

    # Required columns
    required_cols = {"id", "question", "answers"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing expected columns: {required_cols - set(df.columns)}")

    print(f"✅ Loaded {len(df)} rows")

    # Optional sampling
    if CONFIG["eval_limit"]:
        df = df.sample(n=CONFIG["eval_limit"], random_state=42)
        print(f"🔹 Sampling {CONFIG['eval_limit']} examples")

    # JSONL export
    with open(out_path, "w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Q/A pairs"):
            answers = row["answers"]

            if isinstance(answers, dict) and "text" in answers and len(answers["text"]) > 0:
                answer = answers["text"][0]
            else:
                continue

            record = {
                "id": row["id"],
                "question": row["question"],
                "answer": answer,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Saved base eval set → {out_path}")
    return out_path


# --------------------------------------------
if __name__ == "__main__":
    build_eval_base()
