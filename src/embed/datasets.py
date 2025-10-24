from pathlib import Path
from datasets import load_dataset, Dataset
from src.rag.setting import settings

def _ensure_text_field(ds: Dataset, field: str):
    if field not in ds.column_names:
        raise ValueError(f"Field '{field}' not found. Columns: {ds.column_names}")

def load_hf_dataset() -> Dataset:
    """Charge le dataset Hugging Face selon le .env."""
    name, split = settings.HF_DATASET_NAME, settings.HF_DATASET_SPLIT
    print(f"🔹 Loading HF dataset: {name} [{split}]")
    ds = load_dataset(name, split=split)
    _ensure_text_field(ds, settings.TEXT_FIELD)
    return ds

def save_dataset_parquet(ds: Dataset, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"💾 Saving parquet → {out_path}")
    ds.to_parquet(str(out_path))
    print(f"✅ Saved {len(ds)} rows")

def ingest_raw_to_parquet() -> Path:
    """Ingestion étape 1 : sauvegarde brute en Parquet dans data/raw/."""
    settings.ensure_dirs()
    out_path = settings.RAW_DIR / f"{settings.HF_DATASET_NAME}_{settings.HF_DATASET_SPLIT}.parquet"
    if out_path.exists():
        print(f"✅ Already exists: {out_path}")
        return out_path
    ds = load_hf_dataset()
    save_dataset_parquet(ds, out_path)
    return out_path
