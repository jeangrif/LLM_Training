from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm

class TextChunker:
    """Gère la découpe de texte et la génération de chunks à partir d'un parquet."""

    def __init__(self, chunk_size: int, overlap: int, text_field: str):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.text_field = text_field

    def chunk_text(self, text: str):
        if not text:
            return []
        chunks, start = [], 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.overlap
        return chunks

    def make_chunks(self, parquet_path: Path, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "chunks.jsonl"

        df = pd.read_parquet(parquet_path)
        if self.text_field not in df.columns:
            raise ValueError(f"Column '{self.text_field}' not found in dataset")

        before = len(df)
        df = df.drop_duplicates(subset=[self.text_field]).reset_index(drop=True)
        after = len(df)
        print(f"🧹 Removed {before - after} duplicate contexts → {after} unique rows remain.")

        print(f"🔹 Creating chunks from {after} unique rows...")
        with open(out_path, "w", encoding="utf-8") as f_out:
            for i, text in enumerate(tqdm(df[self.text_field], desc="Chunking")):
                for j, chunk in enumerate(self.chunk_text(text)):
                    record = {"source_id": i, "chunk_id": j, "text": chunk.strip()}
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"✅ Saved chunks to {out_path}")
        return out_path
