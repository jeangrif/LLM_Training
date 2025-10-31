import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Generate embeddings from text chunks stored in a JSONL file.
    """

    # Initialize the embedder with model name and batch size.
    # Loads the SentenceTransformer model into memory.
    def __init__(self, model_name: str, batch_size: int):
        self.model_name = model_name
        self.batch_size = batch_size
        print(f"🔹 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    @staticmethod
    # Read text chunks sequentially from a JSONL file.
    # Returns a generator yielding parsed JSON records.
    def load_chunks(chunks_path: Path):
        """Lit les chunks depuis le JSONL."""
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

    def encode_chunks(self, chunks_path: Path, out_dir: Path):
        """
        Encode all text chunks into vector embeddings and save them as a NumPy array.

        Args:
            chunks_path: Path to the JSONL file containing text chunks.
            out_dir: Directory where embeddings.npy will be saved.

        Returns:
            Path to the saved embeddings file.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        emb_path = out_dir / "embeddings.npy"

        chunks = [rec["text"] for rec in self.load_chunks(chunks_path)]
        print(f"🔹 Encoding {len(chunks)} chunks in batches of {self.batch_size}...")

        # Encode all chunks in batches using the embedding model.
        # Normalization ensures consistent vector magnitude across embeddings.
        embeddings = self.model.encode(
            chunks,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = np.array(embeddings, dtype=np.float32)
        np.save(emb_path, embeddings)
        print(f"💾 Embeddings saved → {emb_path}")
        print(f"✅ Done. Shape = {embeddings.shape}")
        return emb_path
