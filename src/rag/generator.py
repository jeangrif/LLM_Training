# src/rag/runner.py
from src.models.llama_cpp import LlamaCppProvider


class RagGenerator:
    """Assemble le contexte et génère une réponse avec le modèle local."""

    def __init__(self, model_meta, model_cfg):
        self.model = LlamaCppProvider(model_meta, model_cfg=model_cfg)

    def build_prompt(self, query: str, contexts: list[str]) -> str:
        """Construit un prompt simple et clair pour le modèle."""
        context_text = "\n\n".join(contexts)
        prompt = (
            "You are an intelligent assistant. Use the provided context to answer the question.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        return prompt

    def generate(self, query: str, contexts: list[str]) -> str:
        """Génère la réponse complète à partir du contexte."""
        prompt = self.build_prompt(query, contexts)
        return self.model.generate(prompt)

    def close(self):
        """Libère explicitement les ressources du modèle."""
        if hasattr(self.model, "close"):
            self.model.close()
