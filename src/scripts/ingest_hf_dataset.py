#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: Ingest a Hugging Face dataset and save it as a Parquet file in data/raw/.

This script:
  - downloads the dataset from Hugging Face Hub
  - ensures the target text field exists
  - saves the dataset in Parquet format for later use
"""

from pathlib import Path
from datasets import load_dataset, Dataset

# --------------------------------------------
# ⚙️ Configuration (edit here as needed)
# --------------------------------------------
CONFIG = {
    "hf_dataset_name": "squad",       # Hugging Face dataset name
    "hf_dataset_split": "train",      # Split to load ("train", "validation", etc.)
    "text_field": "context",          # Field containing the text passages
    "raw_dir": Path("data/raw"),      # Output directory for parquet files
}

# --------------------------------------------
# 🔍 Utility functions
# --------------------------------------------
def _ensure_text_field(ds: Dataset, field: str):
    """Check that the dataset contains the required text field."""
    if field not in ds.column_names:
        raise ValueError(f"❌ Field '{field}' not found. Available columns: {ds.column_names}")


# --------------------------------------------
# 📦 Dataset loading & saving
# --------------------------------------------
def load_hf_dataset() -> Dataset:
    """Load the Hugging Face dataset based on the configuration."""
    name, split = CONFIG["hf_dataset_name"], CONFIG["hf_dataset_split"]
    print(f"🔹 Loading HF dataset: {name} [{split}]")
    ds = load_dataset(name, split=split)
    _ensure_text_field(ds, CONFIG["text_field"])
    return ds


def save_dataset_parquet(ds: Dataset, out_path: Path):
    """Save the dataset as a Parquet file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"💾 Saving parquet → {out_path}")
    ds.to_parquet(str(out_path))
    print(f"✅ Saved {len(ds)} rows to {out_path}")


# --------------------------------------------
# 🚀 Main ingestion logic
# --------------------------------------------
def ingest_raw_to_parquet() -> Path:
    """Main entrypoint: load, check, and save dataset as Parquet."""
    out_path = CONFIG["raw_dir"] / f"{CONFIG['hf_dataset_name']}_{CONFIG['hf_dataset_split']}.parquet"

    if out_path.exists():
        print(f"✅ Already exists: {out_path}")
        return out_path

    ds = load_hf_dataset()
    save_dataset_parquet(ds, out_path)
    return out_path


# --------------------------------------------
if __name__ == "__main__":
    ingest_raw_to_parquet()

