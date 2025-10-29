#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end data preparation script.

This script orchestrates the 3 preparation steps:
1. Download & save HF dataset to Parquet
2. Extract evaluation base questions
3. Generate paraphrased questions for augmentation
"""

from pathlib import Path

# Import the three main steps
from ingest_hf_dataset import ingest_raw_to_parquet
from build_eval_base import build_eval_base
from generate_paraphrases import generate_paraphrases


def main():
    print("\n🚀 Starting full data preparation pipeline\n" + "-" * 60)

    # Step 1 – Ingest HF dataset
    print("\n[1/3] 📦 Ingesting Hugging Face dataset → Parquet")
    parquet_path = ingest_raw_to_parquet()

    # Step 2 – Build base evaluation set
    print("\n[2/3] 🧩 Building base evaluation dataset (id, question, answer)")
    base_path = build_eval_base()

    # Step 3 – Generate paraphrased variations
    print("\n[3/3] 💬 Generating paraphrased question variations (5 levels)")
    aug_path = generate_paraphrases()

    print("\n✅ All data preparation steps completed successfully!")
    print(f"   ├── Raw dataset:      {parquet_path}")
    print(f"   ├── Base eval set:    {base_path}")
    print(f"   └── Augmented eval:   {aug_path}")
    print("-" * 60)


if __name__ == "__main__":
    main()

