import os

def load_llm_provider():
    name = os.getenv("RAGBOT_LLM_PROVIDER", "llama_cpp").lower()
    if name == "llama_cpp":
        from .llama_cpp import LlamaCppProvider
        return LlamaCppProvider()
    if name == "hf_local":
        from .hf_local import HFLocalProvider
        return HFLocalProvider()
    raise ValueError(f"Unknown provider: {name}")
