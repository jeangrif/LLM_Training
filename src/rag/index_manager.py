import json
import faiss
from datetime import datetime
from pathlib import Path
from src.embed.chunking import make_chunks
from src.embed.embeddings import generate_embeddings
from src.embed.faiss_index import build_faiss_index
from src.rag.setting import RagSettings


class IndexManager:
    def __init__(self, embed_settings, parquet_path: str = None):
        self.embed_cfg = embed_settings
        self.parquet_path = Path(parquet_path or self.embed_cfg.parquet_path)

        # --- Preprocess Hydra native ---
        RagSettings(self.embed_cfg).ensure_dirs()

        # ---Embeddings and chunking parameters ---
        self.embedding_model = self.embed_cfg.embedding_model
        self.chunk_size = int(self.embed_cfg.chunk_size)
        self.chunk_overlap = int(self.embed_cfg.chunk_overlap)

        # --- Folder structure ---
        self.base_dir = Path(self.embed_cfg.index_dir)
        self.model_name = self.embedding_model.replace("/", "-")

        self.index_dir = self.base_dir / f"{self.model_name}__chunk{self.chunk_size}_ov{self.chunk_overlap}"
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # --- main files ---
        self.faiss_path = self.index_dir / "faiss.index"
        self.docs_path = self.index_dir / "docs.jsonl"
        self.meta_path = self.index_dir / "metadata.json"

    # ---------------------------------------------------------
    def run(self, **kwargs):
        """Interface unifiée pour orchestrateur Hydra."""
        if self._is_ready():
            print(f"✅ Using cached index → {self.index_dir}")
            return {"status": "cached", "index_dir": str(self.index_dir)}

        print(f"⚙️ Building FAISS index for {self.model_name} (chunk={self.chunk_size}, overlap={self.chunk_overlap})")
        index, docs = self._build_from_parquet(self.parquet_path)
        print(f"✅ New index created → {self.index_dir}")
        return {"status": "built", "index_dir": str(self.index_dir), "num_docs": len(docs)}

    # ---------------------------------------------------------
    def _is_ready(self):
        """Vérifie si l’index existe déjà et est complet."""
        return all(p.exists() for p in [self.faiss_path, self.docs_path, self.meta_path])

    def _load_index(self):
        """Charge un index FAISS et les documents correspondants."""
        index = faiss.read_index(str(self.faiss_path))
        with open(self.docs_path, "r", encoding="utf-8") as f:
            docs = [json.loads(line) for line in f]
        return index, docs

    def _build_from_parquet(self, parquet_path: Path):
        """Construit un nouvel index à partir du fichier parquet."""
        chunks_path = make_chunks(
            parquet_path=parquet_path,
            out_dir=self.index_dir,
            text_field=self.embed_cfg.text_field,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )

        # 2️⃣ Embeddings Generation
        emb_path = generate_embeddings(
            chunks_path=chunks_path,
            model_name=self.embedding_model,
            batch_size=self.embed_cfg.embedding_batch_size,
            out_dir=self.index_dir,
        )

        # 3️⃣ Construct Index FAISS
        faiss_path, docs_path = build_faiss_index(
            embeddings_path=emb_path,
            chunks_path=chunks_path,
            index_dir=self.index_dir,
        )

        # --- Clean temporary file ---
        for tmp_file in [chunks_path, emb_path]:
            try:
                if tmp_file.exists():
                    tmp_file.unlink()
                    print(f"🧹 Deleted temporary file {tmp_file}")
            except Exception as e:
                print(f"⚠️ Could not delete {tmp_file}: {e}")

        # --- Save metadata ---
        metadata = {
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "num_chunks": sum(1 for _ in open(docs_path, "r")),
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"💾 Metadata saved → {self.meta_path}")

        return self._load_index()
