# src/rag/retriever/dense_faiss.py

import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


class FaissRetriever:
    def __init__(self, index_dir: Path = None, parquet_path: Path = None, model_name = None):
        """
        Args:
            index_dir (Path): dossier contenant faiss.index, docs.jsonl, metadata.json
            parquet_path (Path): chemin du dataset source (pour build auto si pas d'index)
        """
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.index, self.docs = self._load_from_dir(index_dir)

        # Info interne
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.retrieval_type = "dense"
        self.index_type = "faiss_flat"

    # ----------------------------------------------------------
    def _load_from_dir(self, index_dir: Path):
        index_dir = Path(index_dir)
        index = faiss.read_index(str(index_dir / "faiss.index"))
        with open(index_dir / "docs.jsonl", "r", encoding="utf-8") as f:
            docs = [json.loads(line) for line in f]
        return index, docs

    # ----------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5):
        q_emb = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        scores, ids = self.index.search(np.array(q_emb, dtype=np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            doc = self.docs[idx]
            results.append({
                "text": doc["text"],
                "score": float(score),
                "metadata": {k: v for k, v in doc.items() if k != "text"}
            })
        return results
