# src/rag/retriever/dense_faiss.py

import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


class FaissRetriever:
    """
    Dense retriever using a FAISS index and SentenceTransformer embeddings.
    Retrieves the most relevant text chunks based on cosine similarity.
    """

    def __init__(self, index_dir: Path = None, parquet_path: Path = None, model_name=None):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.index, self.docs = self._load_from_dir(index_dir)

        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.retrieval_type = "dense"
        self.index_type = "faiss_flat"

    def _load_from_dir(self, index_dir: Path):
        index_dir = Path(index_dir)
        index = faiss.read_index(str(index_dir / "faiss.index"))
        with open(index_dir / "docs.jsonl", "r", encoding="utf-8") as f:
            docs = [json.loads(line) for line in f]
        return index, docs

    # --- 🔹 Ajout : méthode interne pour générer un doc_id stable ---
    def _doc_id(self, doc: dict) -> str:
        sid = doc.get("source_id")
        cid = doc.get("chunk_id")
        if sid is not None and cid is not None:
            return f"{sid}:{cid}"
        # fallback robuste si pas de metadata
        return f"txt:{hash(doc['text'])}"

    def retrieve(self, query: str, top_k: int = 5):
        q_emb = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        scores, ids = self.index.search(np.array(q_emb, dtype=np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            doc = self.docs[idx]
            results.append({
                "doc_id": self._doc_id(doc),     # ✅ ajouté
                "text": doc["text"],
                "score": float(score),
                "metadata": {k: v for k, v in doc.items() if k != "text"}
            })
        return results

