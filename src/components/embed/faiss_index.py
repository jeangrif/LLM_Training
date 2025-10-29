import json
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm


class FaissIndexBuilder:
    """Construit et sauvegarde un index FAISS à partir d'embeddings et de chunks."""

    def __init__(self):
        self.index = None
        self.dim = None

    @staticmethod
    def load_embeddings(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {path}")
        emb = np.load(path)
        print(f"✅ Loaded embeddings {emb.shape}")
        return emb

    @staticmethod
    def load_chunks(chunks_path: Path):
        """Lit les chunks JSONL avec texte et métadonnées."""
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

    def build(self, embeddings_path: Path, chunks_path: Path, index_dir: Path):
        """
        Construit un index FAISS à partir des embeddings normalisés
        et génère un fichier docs.jsonl aligné (texte + meta).
        """
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss_path = index_dir / "faiss.index"
        docs_path = index_dir / "docs.jsonl"

        embeddings = self.load_embeddings(embeddings_path)
        chunks = list(self.load_chunks(chunks_path))

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings")

        faiss.normalize_L2(embeddings)
        self.dim = embeddings.shape[1]
        print(f"🔹 Building FAISS index (dim={self.dim}, n={len(embeddings)}) ...")

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        faiss.write_index(self.index, str(faiss_path))
        print(f"💾 FAISS index saved → {faiss_path}")

        with open(docs_path, "w", encoding="utf-8") as f:
            for rec in tqdm(chunks, desc="Writing docs"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"💾 Docs metadata saved → {docs_path}")

        if self.index.ntotal != len(chunks):
            raise RuntimeError("FAISS index size mismatch with docs.jsonl")

        print("✅ FAISS build complete.")
        return faiss_path, docs_path
