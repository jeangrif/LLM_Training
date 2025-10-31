import json
from pathlib import Path
import bm25s
import Stemmer
from src.components.retriever.base_retriever import RetrieverBase


class BM25Retriever(RetrieverBase):
    """
    Sparse retriever based on a prebuilt BM25 index (using bm25s).
    Loads the index from disk if available, otherwise rebuilds it from docs.jsonl.
    """

    def __init__(self, index_dir: Path):
        """
        Args:
            index_dir (Path): Path to the directory containing:
                - docs.jsonl
                - bm25_index/ (built by BM25IndexBuilder)
        """
        self.index_dir = Path(index_dir)
        self.docs_path = self.index_dir / "docs.jsonl"
        self.bm25_dir = self.index_dir / "bm25_index"

        if not self.docs_path.exists():
            raise FileNotFoundError(f"Missing docs.jsonl in {self.index_dir}")
        if not self.bm25_dir.exists():
            raise FileNotFoundError(f"Missing BM25 index directory: {self.bm25_dir}")

        # Load the documents
        with open(self.docs_path, "r", encoding="utf-8") as f:
            self.docs = [json.loads(line) for line in f]
        self.texts = [d["text"] for d in self.docs]

        # Load the prebuilt BM25 index from disk
        print(f"📦 Loading BM25 index from {self.bm25_dir} ...")
        self.bm25 = bm25s.BM25.load(str(self.bm25_dir), load_corpus=False)
        self.retrieval_type = "sparse_bm25"

    # -----------------------------------
    def _doc_id(self, doc: dict) -> str:
        sid, cid = doc.get("source_id"), doc.get("chunk_id")
        if sid is not None and cid is not None:
            return f"{sid}:{cid}"
        return f"txt:{hash(doc['text'])}"

    # -----------------------------------
    def retrieve(self, query: str, top_k: int = 5):
        """
        Retrieve top-k chunks matching the query.
        """
        query_tokens = bm25s.tokenize(query, stopwords="en")
        indices, scores = self.bm25.retrieve(query_tokens, k=top_k)

        output = []
        for idx, score in zip(indices[0], scores[0]):
            doc = self.docs[idx]
            output.append({
                "doc_id": self._doc_id(doc),
                "text": doc["text"],
                "score": float(score),
                "metadata": {k: v for k, v in doc.items() if k != "text"}
            })
        return output

    # -----------------------------------
    def get_info(self):
        return {
            "type": self.retrieval_type,
            "num_docs": len(self.docs),
            "index_path": str(self.bm25_dir)
        }
