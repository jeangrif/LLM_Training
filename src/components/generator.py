# src/rag/rag_runner.py
from src.components.llm.llama_cpp import LlamaCppProvider


class RagGenerator:
    """
    Build prompts and generate final answers using a local LLM.
    Integrates with the LlamaCpp provider for text generation.
    """

    # Initialize the generator with a local model provider.
    # Uses model metadata and configuration to load the appropriate LLM instance.
    def __init__(self, model_meta, model_cfg):
        self.model = LlamaCppProvider(model_meta, model_cfg=model_cfg)

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

    def generate(self, query: str, contexts: list[str]) -> str:
        """
        Generate a complete answer from the provided query and context.

        Args:
            query (str): Input question.
            contexts (list[str]): Retrieved or reranked context passages.

        Returns:
            str: Model-generated answer text.
        """
        if hasattr(self.model, "reset"):
            self.model.reset()
        prompt = self.build_prompt(query, contexts)
        return self.model.generate(prompt)

    def close(self):
        """
        Cleanly release model resources (e.g., GPU memory or file handles).
        """
        if hasattr(self.model, "close"):
            self.model.close()
