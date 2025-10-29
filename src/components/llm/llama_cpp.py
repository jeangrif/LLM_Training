import os
import sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from llama_cpp import Llama
from pathlib import Path
from .base_model import BaseProvider
from platformdirs import user_cache_dir
from src.utils.cache import cache_llm_if_needed

def _models_dir() -> Path:
    root = Path(user_cache_dir("LLMTraining")).resolve() / "llm"
    root.mkdir(parents=True, exist_ok=True)
    return root
@contextmanager
def suppress_all_output():
    """Masque tous les logs C/C++ (stdout + stderr) même pendant la libération mémoire."""
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
    def __init__(self, model_meta: dict, model_cfg: dict):
        super().__init__("llama_cpp")

        # --- Vérification de cohérence ---
        if not model_meta or "llm_path" not in model_meta:
            raise ValueError("❌ Missing model metadata (did you run 'check_models' first?)")

        local_path = Path(model_meta["llm_path"])
        if not local_path.exists():
            raise FileNotFoundError(f"❌ LLM file not found at {local_path}")

        # --- Hyperparamètres Hydra (plus de getenv) ---
        self.n_ctx = model_cfg.get("n_ctx", 8192)
        self.n_threads = model_cfg.get("n_threads", 8)
        self.n_gpu_layers = model_cfg.get("n_gpu_layers", 32)
        self.max_new = model_cfg.get("max_new_tokens", 512)
        self.temperature = model_cfg.get("temperature", 0.7)
        self.top_p = model_cfg.get("top_p", 0.95)
        self.chat_format = model_meta.get("chat_format", "mistral-instruct")

        print(f"🧠 Loading {model_meta['llm_repo']} ({local_path.name})...")
        print(f"⚙️ Config: ctx={self.n_ctx}, threads={self.n_threads}, gpu_layers={self.n_gpu_layers}")

        # --- Chargement silencieux ---
        with suppress_all_output():
            self.llm = Llama(
                model_path=str(local_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                chat_format=self.chat_format,
            )

        # --- Métadonnées pour EvalLogger ---
        self._model_file = local_path.name
        self.messages = []
        print(f"✅ Llama.cpp model ready → {local_path}")

    def generate(self, prompt: str, **kwargs) -> str:
        max_tokens = int(kwargs.get("max_new_tokens", self.max_new))
        self.messages.append({"role": "user", "content": prompt})

        # 🔇 Silence aussi pendant la génération + cleanup
        with suppress_all_output():
            out = self.llm.create_chat_completion(
                messages=self.messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=False,
            )

        response = out["choices"][0]["message"]["content"].strip()
        self.messages.append({"role": "assistant", "content": response})
        return response

    def reset(self):
        self.messages = []

    def get_model_info(self):
        return {
            "provider": "llama_cpp",
            "model_file": self._model_file,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "n_gpu_layers": self.n_gpu_layers,
        }

    def close(self):
        """Ferme proprement le modèle et libère silencieusement la mémoire GPU."""
        if hasattr(self, "llm"):
            with suppress_all_output():
                del self.llm
            import gc
            gc.collect()
