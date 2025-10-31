import json
from pathlib import Path
from datetime import datetime
import bm25s
import Stemmer


class BM25IndexBuilder:
    """
    Build and save a BM25 index using the bm25s library.
    Compatible with the same output structure as FAISS indexes.
    """

    def __init__(self, stopwords: str = "en", stemmer_lang: str = "english"):
        """
        Args:
            stopwords (str): Language code for stopword removal.
            stemmer_lang (str): Language for optional stemming ("english", "french", etc.)
        """
        self.stopwords = stopwords
        self.stemmer_lang = stemmer_lang

    # ---------------------------------------------------------
    def build_index(self, docs_path: Path, out_dir: Path):
        """
        Build (and optionally cache) a BM25 index from docs.jsonl.

        Args:
            docs_path (Path): Path to docs.jsonl (chunked texts).
            out_dir (Path): Directory where the BM25 index will be saved.

        Returns:
            Path to the BM25 index directory.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        bm25_dir = out_dir / "bm25_index"
        bm25_dir.mkdir(parents=True, exist_ok=True)
        meta_path = bm25_dir / "bm25.meta.json"

        # Skip rebuild if already exists
        if meta_path.exists():
            print(f"✅ Using existing BM25 index at {bm25_dir}")
            return bm25_dir

        print(f"⚙️ Building BM25 index from {docs_path.name} → {bm25_dir}")

        # 1️⃣ Load corpus
        with open(docs_path, "r", encoding="utf-8") as f:
            docs = [json.loads(line) for line in f]
        texts = [d["text"] for d in docs]

        # 2️⃣ Tokenization
        stemmer = Stemmer.Stemmer(self.stemmer_lang)
        tokens = bm25s.tokenize(texts, stopwords=self.stopwords, stemmer=stemmer)

        # 3️⃣ Index building
        retriever = bm25s.BM25()
        retriever.index(tokens)

        # 4️⃣ Try saving index (if version supports it)
        try:
            retriever.save(str(bm25_dir), corpus=texts)
            print(f"💾 BM25 index saved to {bm25_dir}")
        except Exception as e:
            # fallback for bm25s<1.0 (no .save/.load)
            print(f"⚠️ Could not save BM25 index (bm25s<1.0) → {e}")
            with open(bm25_dir / "corpus.json", "w", encoding="utf-8") as f:
                json.dump(texts, f, ensure_ascii=False, indent=2)

        # 5️⃣ Metadata
        meta = {
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "backend": "bm25s",
            "stopwords": self.stopwords,
            "stemmer": self.stemmer_lang,
            "num_docs": len(texts),
            "source_docs": str(docs_path),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"✅ BM25 metadata saved to {meta_path}")

        return bm25_dir

    # ---------------------------------------------------------
    @staticmethod
    def load_index(index_dir: Path):
        """
        Load an existing BM25 index (if saved), else return None.
        """
        meta_path = Path(index_dir) / "bm25.meta.json"
        if not meta_path.exists():
            print(f"⚠️ No BM25 metadata found at {meta_path}")
            return None

        try:
            retriever = bm25s.BM25.load(str(index_dir), load_corpus=True)
            print(f"✅ BM25 index loaded from {index_dir}")
            return retriever
        except Exception as e:
            print(f"⚠️ Could not load BM25 index: {e}")
            return None
