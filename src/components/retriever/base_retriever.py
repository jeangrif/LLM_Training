# src/rag/retriever/base.py
from abc import ABC, abstractmethod

class RetrieverBase(ABC):
    """Interface commune à tous les retrievers."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5):
        """Retourne une liste de documents (dict) avec score et texte."""
        pass

    @abstractmethod
    def get_info(self):
        """Retourne les métadonnées du retriever (pour logs)."""
        pass
