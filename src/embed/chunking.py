import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def chunk_text(text: str, size: int, overlap: int):
    """Découpe un texte long en morceaux avec chevauchement."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def make_chunks(parquet_path: Path, out_dir: Path, text_field: str, chunk_size: int, overlap: int):
    """
    Lit le dataset parquet et génère un fichier JSONL de chunks uniques.
    Args:
        parquet_path: chemin vers le dataset .parquet
        out_dir: dossier où enregistrer les chunks
        text_field: colonne contenant le texte
        chunk_size: longueur des segments
        overlap: chevauchement entre deux chunks
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chunks.jsonl"

    # 🔹 Lecture du dataset
    df = pd.read_parquet(parquet_path)
    if text_field not in df.columns:
        raise ValueError(f"Column '{text_field}' not found in dataset")

    # 🔹 Supprimer les doublons de contexte
    before = len(df)
    df = df.drop_duplicates(subset=[text_field]).reset_index(drop=True)
    after = len(df)
    print(f"🧹 Removed {before - after} duplicate contexts → {after} unique rows remain.")

    # 🔹 Génération des chunks
    print(f"🔹 Creating chunks from {after} unique rows...")
    with open(out_path, "w", encoding="utf-8") as f_out:
        for i, text in enumerate(tqdm(df[text_field], desc="Chunking")):
            for j, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
                record = {"source_id": i, "chunk_id": j, "text": chunk.strip()}
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Saved chunks to {out_path}")
    return out_path
