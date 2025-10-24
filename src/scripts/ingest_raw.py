from pathlib import Path
import os
import sys

# Force le CWD à la racine du projet
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.embed.datasets import ingest_raw_to_parquet

if __name__ == "__main__":
    out = ingest_raw_to_parquet()
    print(f"\n📦 Dataset parquet: {out}")
