import os
import sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from llama_cpp import Llama
from pathlib import Path
from .base_model import BaseProvider
from platformdirs import user_cache_dir
import gc
from src.utils.cache import cache_llm_if_needed

# Return the default local directory for storing LLM model files.
# Creates the directory if it does not already exist.
def _models_dir() -> Path:
    root = Path(user_cache_dir("LLMTraining")).resolve() / "llm"
    root.mkdir(parents=True, exist_ok=True)
    return root
@contextmanager
def suppress_all_output():
    """
    Context manager that suppresses all stdout and stderr output,
    including C/C++ logs during model loading or memory release.
    """
    devnull = open(os.devnull, "w")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()


class LlamaCppProvider(BaseProvider):
    """
    Wrapper for local LLaMA models (via llama.cpp).
    """

    def __init__(self, model_meta, model_cfg):
        super().__init__("llama_cpp")

        path = Path(model_meta["llm_path"])
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self.n_ctx = model_cfg.get("n_ctx", 8192)
        self.max_new = model_cfg.get("max_new_tokens", 512)
        self.temperature = model_cfg.get("temperature", 0.7)
        self.top_p = model_cfg.get("top_p", 0.95)
        self.n_threads = model_cfg.get("n_threads", 8)
        self.n_gpu_layers = model_cfg.get("n_gpu_layers", 32)
        self.chat_format = model_meta.get("chat_format", "mistral-instruct")

        print(f"🧠 Loading model from {path}...")
        with suppress_all_output():
            self.llm = Llama(
                model_path=str(path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                chat_format=self.chat_format,
            )

        self.messages = []
        print("✅ Model ready.")

    def generate(self, prompt: str) -> str:
        """Stateless generation."""
        with suppress_all_output():
            result = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_new,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        return result["choices"][0]["message"]["content"].strip()

    def chat(self, user_message: str) -> str:
        """Stateful chat."""

        self.messages.append({"role": "user", "content": user_message})
        print("DEBUG : Il comprend bien qu'on est en mode statefull", self.messages)
        with suppress_all_output():
            result = self.llm.create_chat_completion(
                messages=self.messages,
                max_tokens=self.max_new,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        response = result["choices"][0]["message"]["content"].strip()
        self.messages.append({"role": "assistant", "content": response})
        return response

    def chat_ephemeral(self, user_message: str, system_context: str) -> str:
        """
        Fait une complétion en utilisant l'historique + un contexte système éphémère,
        mais n'enregistre dans l'historique que le tour (user, assistant) "propre",
        sans le bloc de contexte.
        """
        msgs = list(self.messages)  # copie
        if system_context:
            msgs.append({"role": "system", "content": system_context})
        msgs.append({"role": "user", "content": user_message})

        with suppress_all_output():
            result = self.llm.create_chat_completion(
                messages=msgs,
                max_tokens=self.max_new,
                temperature=self.temperature,
                top_p=self.top_p,
            )

        response = result["choices"][0]["message"]["content"].strip()

        # On ne persiste que la conversation "propre"
        self.messages.append({"role": "user", "content": user_message})
        self.messages.append({"role": "assistant", "content": response})
        return response

    def reset(self):
        self.messages = []

    def close(self):
        if hasattr(self, "llm"):
            with suppress_all_output():
                del self.llm
            gc.collect()

