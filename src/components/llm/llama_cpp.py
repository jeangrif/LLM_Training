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
    Provider wrapper for running local LLaMA models through llama.cpp.
    Handles model loading, prompt generation, and resource cleanup.
    """

    # Initialize the llama.cpp provider with model metadata and configuration parameters.
    # Validates paths, loads the model silently, and prepares runtime settings.
    def __init__(self, model_meta: dict, model_cfg: dict):
        super().__init__("llama_cpp")

        # Validate that model metadata and local file path are correctly defined before loading.
        if not model_meta or "llm_path" not in model_meta:
            raise ValueError("❌ Missing model metadata (did you run 'check_models' first?)")

        local_path = Path(model_meta["llm_path"])
        if not local_path.exists():
            raise FileNotFoundError(f"❌ LLM file not found at {local_path}")

        # Load runtime parameters (context size, threads, GPU layers, etc.) from the Hydra configuration.
        self.n_ctx = model_cfg.get("n_ctx", 8192)
        self.n_threads = model_cfg.get("n_threads", 8)
        self.n_gpu_layers = model_cfg.get("n_gpu_layers", 32)
        self.max_new = model_cfg.get("max_new_tokens", 512)
        self.temperature = model_cfg.get("temperature", 0.7)
        self.top_p = model_cfg.get("top_p", 0.95)
        self.chat_format = model_meta.get("chat_format", "mistral-instruct")

        print(f"🧠 Loading {model_meta['llm_repo']} ({local_path.name})...")
        print(f"⚙️ Config: ctx={self.n_ctx}, threads={self.n_threads}, gpu_layers={self.n_gpu_layers}")

        # Load the LLaMA model while suppressing all underlying C++ logs.
        with suppress_all_output():
            self.llm = Llama(
                model_path=str(local_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                chat_format=self.chat_format,
            )


        self._model_file = local_path.name
        self.messages = []
        print(f"✅ Llama.cpp model ready → {local_path}")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a model response for a given input prompt using llama.cpp.

        Args:
            prompt (str): User prompt including context and question.
            **kwargs: Optional generation parameters such as max_new_tokens.

        Returns:
            str: Generated text output from the model.
        """
        max_tokens = int(kwargs.get("max_new_tokens", self.max_new))
        self.messages.append({"role": "user", "content": prompt})

        # Suppress verbose C++ logs during inference for cleaner output.
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

    # Reset the stored chat history for a fresh conversation state.
    def reset(self):
        self.messages = []

    # Return basic model configuration and runtime details for logging or evaluation.
    def get_model_info(self):
        return {
            "provider": "llama_cpp",
            "model_file": self._model_file,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "n_gpu_layers": self.n_gpu_layers,
        }

    def close(self):
        """
        Cleanly release the LLaMA model and free GPU memory silently.
        """
        if hasattr(self, "llm"):
            with suppress_all_output():
                del self.llm

            # Force garbage collection to ensure GPU and memory resources are fully released.
            gc.collect()
