import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer


def load_chunks(chunks_path: Path):
    """Lit les chunks depuis le JSONL."""
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def generate_embeddings(chunks_path: Path, model_name: str, batch_size: int, out_dir: Path):
    """
    Génère les embeddings à partir des chunks et les sauvegarde dans embeddings.npy.
    Args:
        chunks_path: chemin vers chunks.jsonl
        model_name: nom du modèle SentenceTransformer
        batch_size: taille des batchs d’encodage
        out_dir: dossier où sauvegarder embeddings.npy
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "embeddings.npy"

    print(f"🔹 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    chunks = [rec["text"] for rec in load_chunks(chunks_path)]
    print(f"🔹 Encoding {len(chunks)} chunks in batches of {batch_size}...")
    embeddings = model.encode(
        chunks,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.array(embeddings, dtype=np.float32)
    np.save(emb_path, embeddings)
    print(f"💾 Embeddings saved → {emb_path}")
    print(f"✅ Done. Shape = {embeddings.shape}")
    return emb_path

