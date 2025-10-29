#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: Generate 5 levels of paraphrased questions for evaluation.
"""

import json
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------------------------
# ⚙️ Configuration (edit here as needed)
# --------------------------------------------
CONFIG = {
    "model_name": "eugenesiow/bart-paraphrase",   # Paraphrasing model
    "input_path": Path("data/eval/base_questions.jsonl"),   # Base dataset
    "output_path": Path("data/eval/augmented_questions.jsonl"),  # Output file
    "limit": 200,         # number of examples to process (None = all)
    "offset": 0,          # skip first N examples
    "max_new_tokens": 64, # max tokens for generation
    "device_preference": "auto",  # auto, cpu, cuda, or mps
}

# --------------------------------------------
# 💬 Paraphrasing hints per level
# --------------------------------------------
VARIATION_HINTS = {
    1: "no change",
    2: "minor paraphrase",
    3: "different wording",
    4: "change sentence structure but same meaning",
    5: "creative reformulation keeping the same intent"
}

# --------------------------------------------
# 🧠 Paraphrase function
# --------------------------------------------
def paraphrase_question(model, tokenizer, question: str, degree: int, device: str, max_new_tokens: int = 64):
    """Generate a paraphrased version of a question given the degree of variation."""
    if degree == 1:
        return question.strip()  # direct copy for level 1

    prompt = f"Paraphrase the following question with {VARIATION_HINTS[degree]}:\n\n{question}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_p=0.95,
        do_sample=True,
        num_beams=3
    )
    paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased.strip()

# --------------------------------------------
# 🚀 Main function
# --------------------------------------------
def generate_paraphrases():
    """Main entrypoint: load base questions, generate paraphrases, and save JSONL output."""
    cfg = CONFIG
    input_path, output_path = cfg["input_path"], cfg["output_path"]

    print(f"🔹 Loading base questions from: {input_path}")
    lines = [json.loads(l) for l in open(input_path, encoding="utf-8")]
    total = len(lines)

    # Apply offset and limit
    if cfg["offset"]:
        lines = lines[cfg["offset"]:]
    if cfg["limit"]:
        lines = lines[:cfg["limit"]]
    print(f"✅ Selected {len(lines)} examples (from {total})")

    # Determine device
    device_pref = cfg["device_preference"].lower()
    if device_pref == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_pref
    print(f"💻 Using device: {device.upper()}")

    # Load model and tokenizer
    print(f"🧠 Loading model: {cfg['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"]).to(device)
    model.eval()

    # Generate paraphrases
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for row in tqdm(lines, desc="Generating paraphrases"):
            qid = row["id"]
            question = row["question"]
            answer = row["answer"]

            for degree in range(1, 6):
                paraphrased = paraphrase_question(
                    model, tokenizer, question, degree, device, cfg["max_new_tokens"]
                )
                record = {
                    "orig_id": qid,
                    "degree": degree,
                    "question": paraphrased,
                    "answer": answer,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Paraphrased dataset saved → {output_path}")
    return output_path


# --------------------------------------------
if __name__ == "__main__":
    generate_paraphrases()
