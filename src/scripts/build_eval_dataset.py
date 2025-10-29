import os
import sys
import json
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

# --------------------------------------------
# 🔧 Project root setup
# --------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Load environment variables
load_dotenv()

# --------------------------------------------
# 💬 Paraphrasing degrees
# --------------------------------------------
# -*- coding: utf-8 -*-
"""
Script to generate 5-level paraphrase variations for evaluation dataset.
Fully configured via .env or environment variables.
"""
import os
import sys
import json
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

# --------------------------------------------
# 🔧 Project root setup
# --------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Load environment variables
load_dotenv()

# --------------------------------------------
# 💬 Paraphrasing degrees
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
def paraphrase_question(model, tokenizer, question: str, degree: int, device: str = "cpu", max_new_tokens: int = 64):
    """Generate a paraphrased version of the question given the degree of variation."""
    if degree == 1:
        return question.strip()  # 🔹 direct copy for degree 1

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
# 🚀 Main logic
# --------------------------------------------
def main():
    # --------------------------------------------
    # 🧩 Load config from .env
    # --------------------------------------------
    model_name = os.getenv("PARAPHRASE_MODEL", "eugenesiow/bart-paraphrase")
    input_path = Path(os.getenv("EVAL_INPUT", "data/eval/base_questions.jsonl"))
    output_path = Path(os.getenv("EVAL_OUTPUT", "data/eval/augmented_questions.jsonl"))
    limit = int(os.getenv("EVAL_LIMIT", 200))
    offset = int(os.getenv("EVAL_OFFSET", 0))
    max_new_tokens = int(os.getenv("EVAL_MAX_NEW_TOKENS", 64))
    device_pref = os.getenv("EVAL_DEVICE", "auto").lower()

    # --------------------------------------------
    # ⚙️ Load dataset
    # --------------------------------------------
    print(f"🔹 Loading {input_path}")
    lines = [json.loads(l) for l in open(input_path, encoding="utf-8")]
    total = len(lines)

    if offset:
        lines = lines[offset:]
    if limit:
        lines = lines[:limit]
    print(f"✅ Selected {len(lines)} examples (from {total})")

    # --------------------------------------------
    # ⚡ Device setup
    # --------------------------------------------
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

    # --------------------------------------------
    # 🧠 Load model
    # --------------------------------------------
    print(f"🧠 Loading model {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    # --------------------------------------------
    # 🪄 Paraphrase generation
    # --------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for row in tqdm(lines, desc="Generating paraphrases"):
            qid = row["id"]
            question = row["question"]
            answer = row["answer"]

            for degree in range(1, 6):
                paraphrased = paraphrase_question(model, tokenizer, question, degree, device, max_new_tokens)
                record = {
                    "orig_id": qid,
                    "degree": degree,
                    "question": paraphrased,
                    "answer": answer,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Paraphrased dataset saved to {output_path}")

# --------------------------------------------
if __name__ == "__main__":
    main()


# --------------------------------------------
# 🧠 Paraphrase function
# --------------------------------------------
def paraphrase_question(model, tokenizer, question: str, degree: int, device: str = "cpu", max_new_tokens: int = 64):
    """Generate a paraphrased version of the question given the degree of variation."""
    if degree == 1:
        return question.strip()  # 🔹 direct copy for degree 1

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
# 🚀 Main logic
# --------------------------------------------
def main():
    # --------------------------------------------
    # 🧩 Load config from .env
    # --------------------------------------------
    model_name = os.getenv("PARAPHRASE_MODEL", "eugenesiow/bart-paraphrase")
    input_path = Path(os.getenv("EVAL_INPUT", "data/eval/base_questions.jsonl"))
    output_path = Path(os.getenv("EVAL_OUTPUT", "data/eval/augmented_questions.jsonl"))
    limit = int(os.getenv("EVAL_LIMIT", 200))
    offset = int(os.getenv("EVAL_OFFSET", 0))
    max_new_tokens = int(os.getenv("EVAL_MAX_NEW_TOKENS", 64))
    device_pref = os.getenv("EVAL_DEVICE", "auto").lower()

    # --------------------------------------------
    # ⚙️ Load dataset
    # --------------------------------------------
    print(f"🔹 Loading {input_path}")
    lines = [json.loads(l) for l in open(input_path, encoding="utf-8")]
    total = len(lines)

    if offset:
        lines = lines[offset:]
    if limit:
        lines = lines[:limit]
    print(f"✅ Selected {len(lines)} examples (from {total})")

    # --------------------------------------------
    # ⚡ Device setup
    # --------------------------------------------
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

    # --------------------------------------------
    # 🧠 Load model
    # --------------------------------------------
    print(f"🧠 Loading model {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    # --------------------------------------------
    # 🪄 Paraphrase generation
    # --------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for row in tqdm(lines, desc="Generating paraphrases"):
            qid = row["id"]
            question = row["question"]
            answer = row["answer"]

            for degree in range(1, 6):
                paraphrased = paraphrase_question(model, tokenizer, question, degree, device, max_new_tokens)
                record = {
                    "orig_id": qid,
                    "degree": degree,
                    "question": paraphrased,
                    "answer": answer,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Paraphrased dataset saved to {output_path}")

# --------------------------------------------
if __name__ == "__main__":
    main()
