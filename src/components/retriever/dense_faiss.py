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

    # Initialize the dense retriever with a pretrained embedding model and a FAISS index.
    # Loads both the index and associated document metadata from the specified directory.
    def __init__(self, index_dir: Path = None, parquet_path: Path = None, model_name = None):
        """
        Args:
            index_dir (Path): Directory containing faiss.index, docs.jsonl, and metadata.json files.
            parquet_path (Path): Optional source dataset path (for auto-building if no index exists).
            model_name (str): Name of the embedding model used for encoding queries.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.index, self.docs = self._load_from_dir(index_dir)

        # Info interne
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.retrieval_type = "dense"
        self.index_type = "faiss_flat"

    # Load an existing FAISS index and its associated documents from the given directory.
    def _load_from_dir(self, index_dir: Path):
        index_dir = Path(index_dir)
        index = faiss.read_index(str(index_dir / "faiss.index"))
        with open(index_dir / "docs.jsonl", "r", encoding="utf-8") as f:
            docs = [json.loads(line) for line in f]
        return index, docs

    # Retrieve the top-k most similar documents for a given query using FAISS.
    # Returns a list of dictionaries containing text, score, and metadata.
    def retrieve(self, query: str, top_k: int = 5):
        # Encode the input query into an embedding vector and normalize it for cosine similarity.
        q_emb = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        scores, ids = self.index.search(np.array(q_emb, dtype=np.float32), top_k)

        results = []
        # Match retrieved vector IDs with corresponding document records and format results.
        for score, idx in zip(scores[0], ids[0]):
            doc = self.docs[idx]
            results.append({
                "text": doc["text"],
                "score": float(score),
                "metadata": {k: v for k, v in doc.items() if k != "text"}
            })
        return results
