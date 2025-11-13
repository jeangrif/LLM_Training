# src/rag/rag_runner.py
from src.components.llm.llama_cpp import LlamaCppProvider
from src.components.generator.prompt_builder import PromptBuilder

class RagGenerator:
    """
    Build prompts and generate final answers using a local LLM.
    Integrates with the LlamaCpp provider for text generation.
    """

    # Initialize the generator with a local model provider.
    # Uses model metadata and configuration to load the appropriate LLM instance.
    def __init__(self, model_meta, model_cfg, stateful=False):
        self.model = LlamaCppProvider(model_meta, model_cfg=model_cfg)
        self.stateful = stateful
        self.prompt_builder = PromptBuilder(
            mode="chat" if self.stateful else "instruct",
            max_contexts=model_cfg.get("max_contexts", 3),
            min_score=model_cfg.get("min_score", 0.0)
        )

    def build_prompt(self, query: str, contexts: list[str]) -> str:
        """
        Construct a simple, structured prompt combining context passages and the user query.

        Args:
            query (str): The input question to answer.
            contexts (list[str]): List of retrieved context strings.

        Returns:
            str: Formatted prompt ready for generation.
        """
        context_text = "\n\n".join(contexts)
        prompt = (
            "You are an intelligent assistant. Use the provided context to answer the question.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        return prompt

    def generate(self, query: str, docs: list[dict]) -> str:
        if self.stateful:
            # ⚠️ nouveau chemin propre : pas de contexte dans l'historique
            system_ctx, user_msg = self.prompt_builder.build_ephemeral(query, docs)
            return self.model.chat_ephemeral(user_msg, system_ctx)
        else:
            # chemin existant inchangé
            prompt = self.prompt_builder.build(query, docs)
            return self.model.generate(prompt)

    def reset(self):
        if hasattr(self.model, "reset"):

            self.model.reset()
    def close(self):
        """
        Cleanly release model resources (e.g., GPU memory or file handles).
        """
        if hasattr(self.model, "close"):
            self.model.close()
