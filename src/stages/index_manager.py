import json
import faiss
from datetime import datetime
from pathlib import Path
from src.components.embed.chunking import TextChunker
from src.components.embed.embeddings import Embedder
from src.components.embed.faiss_index import FaissIndexBuilder
from src.utils.setting import RagSettings



class IndexManager:
    """
    Manage text chunking, embedding generation, and FAISS index creation.
    Checks whether an index matching the current configuration (model, chunk size, overlap)
    already exists; otherwise, builds and saves it.
    """

    # Initialize the index manager with embedding settings and optional parquet input path.
    # Prepares directories, chunker, embedder, and FAISS builder components.
    def __init__(self, embed_settings, parquet_path: str = None):
        self.embed_cfg = embed_settings
        self.parquet_path = Path(parquet_path or self.embed_cfg.parquet_path)

        # Ensure required directories exist according to Hydra configuration.
        RagSettings(self.embed_cfg).ensure_dirs()

        # Load embedding and chunking parameters from the configuration.
        self.embedding_model = self.embed_cfg.embedding_model
        self.chunk_size = int(self.embed_cfg.chunk_size)
        self.chunk_overlap = int(self.embed_cfg.chunk_overlap)

        # Define output directory structure based on model name and chunking parameters.
        self.base_dir = Path(self.embed_cfg.index_dir)
        self.model_name = self.embedding_model.replace("/", "-")

        self.index_dir = self.base_dir / f"{self.model_name}__chunk{self.chunk_size}_ov{self.chunk_overlap}"
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Define paths for FAISS index, documents, and metadata files.
        # Initialize chunker, embedder, and index builder components.
        self.faiss_path = self.index_dir / "faiss.index"
        self.docs_path = self.index_dir / "docs.jsonl"
        self.meta_path = self.index_dir / "metadata.json"
        self.chunker = TextChunker(
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
            text_field=self.embed_cfg.text_field
        )
        self.embedder = Embedder(
            model_name=self.embedding_model,
            batch_size=self.embed_cfg.embedding_batch_size
        )
        self.index_builder = FaissIndexBuilder()

    # ---------------------------------------------------------
    def run(self, **kwargs):
        """
        Unified interface for the Hydra orchestrator.
        Returns the index directory path and build status.
        """
        if self._is_ready():
            print(f"✅ Using cached index → {self.index_dir}")
            return {"status": "cached", "index_dir": str(self.index_dir)}

        print(f"⚙️ Building FAISS index for {self.model_name} (chunk={self.chunk_size}, overlap={self.chunk_overlap})")
        index, docs = self._build_from_parquet(self.parquet_path)
        print(f"✅ New index created → {self.index_dir}")
        return {"status": "built", "index_dir": str(self.index_dir), "num_docs": len(docs)}

    # ---------------------------------------------------------
    def _is_ready(self):
        """
        Check whether the FAISS index and all required files already exist.
        """
        return all(p.exists() for p in [self.faiss_path, self.docs_path, self.meta_path])

    def _load_index(self):
        """
        Load the FAISS index and associated document metadata from disk.
        """
        index = faiss.read_index(str(self.faiss_path))
        with open(self.docs_path, "r", encoding="utf-8") as f:
            docs = [json.loads(line) for line in f]
        return index, docs

    def _build_from_parquet(self, parquet_path: Path):
        """
        Build a new FAISS index from a parquet dataset.
        Performs chunking, embedding generation, FAISS construction,
        and metadata export.
        """
        chunks_path = self.chunker.make_chunks(
            parquet_path=parquet_path,
            out_dir=self.index_dir,
        )

        # 2️⃣ Embeddings Generation
        emb_path = self.embedder.encode_chunks(
            chunks_path=chunks_path,
            out_dir=self.index_dir,
        )

        # 3️⃣ Construct Index FAISS
        faiss_path, docs_path = self.index_builder.build(
            embeddings_path=emb_path,
            chunks_path=chunks_path,
            index_dir=self.index_dir,
        )

        # Remove temporary intermediate files used during index creation.
        for tmp_file in [chunks_path, emb_path]:
            try:
                if tmp_file.exists():
                    tmp_file.unlink()
                    print(f"🧹 Deleted temporary file {tmp_file}")
            except Exception as e:
                print(f"⚠️ Could not delete {tmp_file}: {e}")

        # Save metadata describing the index configuration and build details.
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
