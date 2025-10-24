#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, time, json
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

# importe DIRECTEMENT le provider pour rester simple
from src.models.llama_cpp import LlamaCppProvider

def main() -> int:
    print("=== SMOKE (llama.cpp + 1 .gguf) ===")
    print("[ENV]", json.dumps({
        "HF_MODEL_ID": os.getenv("HF_MODEL_ID"),
        "GGUF_FILENAME": os.getenv("GGUF_FILENAME"),
        "N_GPU_LAYERS": os.getenv("N_GPU_LAYERS"),
        "N_CTX": os.getenv("N_CTX"),
        "N_THREADS": os.getenv("N_THREADS"),
    }, ensure_ascii=False))

    t0 = time.time()
    llm = LlamaCppProvider()
    print(f"[BOOT] {llm.get_model_info()}  ({time.time()-t0:.2f}s)")

    t1 = time.time()
    out = llm.generate("Réponds uniquement par le mot 'OK'.", max_new_tokens=16)
    print(f"[GEN] {out!r}  ({time.time()-t1:.2f}s)")

    ok = out.strip().upper().startswith("OK")
    print("[PASS] ✅" if ok else "[FAIL] ❌")
    return 0 if ok else 2

if __name__ == "__main__":
    raise SystemExit(main())
