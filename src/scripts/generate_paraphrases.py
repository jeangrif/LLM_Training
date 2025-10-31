# -*- coding: utf-8 -*-
"""
Step 3: Generate 5 levels of paraphrased questions for evaluation.
Robust version — 2-stage pipeline:
1. Generate many paraphrase candidates
2. Select 5 progressive levels by semantic similarity
Also generates a full 'initial_questions.jsonl' file with degree=1 for all base questions.
"""
import random
import json
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# --------------------------------------------
# ⚙️ Configuration
# --------------------------------------------
CONFIG = {
    "model_name": "Vamsi/T5_Paraphrase_Paws",  # ✅ more stable paraphraser
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "input_path": Path("data/eval/base_questions.jsonl"),
    "output_path": Path("data/eval/augmented_questions.jsonl"),
    "initial_output_path": Path("data/eval/initial_questions.jsonl"),
    "limit": 200,
    "offset": 0,
    "max_new_tokens": 64,
    "num_candidates": 25,   # number of paraphrases to sample per question
    "device_preference": "auto",
}

# --------------------------------------------
# 🧩 Generate initial dataset
# --------------------------------------------
def generate_initial_dataset(cfg):
    """Create a full dataset copy with degree=1 for all base questions."""
    input_path, output_path = cfg["input_path"], cfg["initial_output_path"]
    print(f"🧩 Generating initial dataset from {input_path}")
    lines = [json.loads(l) for l in open(input_path, encoding="utf-8")]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for row in lines:
            record = {
                "orig_id": row["id"],
                "degree": 1,
                "question": row["question"],
                "answer": row["answer"],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Initial dataset saved → {output_path}")
    return output_path

# --------------------------------------------
# 🧠 Paraphrasing function
# --------------------------------------------
def paraphrase_candidates(model, tokenizer, question, device, max_new_tokens, n=25):
    """Generate multiple paraphrase candidates with sampling."""
    prompt = f"paraphrase: {question}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.9,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        num_return_sequences=n
    )
    texts = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]
    # Clean up repeated prompts or prefixes
    cleaned = [t.replace("Paraphrase:", "").replace("paraphrase:", "").strip() for t in texts]
    return list(set(cleaned))  # remove duplicates

# --------------------------------------------
# 🧮 Select 5 paraphrases progressively by similarity
# --------------------------------------------
def select_progressive_paraphrases(question, candidates, embedder):
    """Select 5 paraphrases evenly spaced by semantic similarity."""
    if not candidates:
        return [question] * 5

    q_emb = embedder.encode(question, convert_to_tensor=True)
    c_emb = embedder.encode(candidates, convert_to_tensor=True)
    sims = util.cos_sim(q_emb, c_emb)[0].cpu().tolist()

    # Sort candidates by similarity descending
    paired = sorted(zip(candidates, sims), key=lambda x: x[1], reverse=True)

    # Remove overly similar (duplicates) and nonsensical (too low similarity)
    filtered = [(c, s) for c, s in paired if 0.6 <= s <= 0.99]
    if len(filtered) < 5:
        filtered = paired  # fallback

    # Evenly sample 5 across similarity range
    step = max(1, len(filtered) // 5)
    selected = [filtered[i][0] for i in range(0, min(len(filtered), step * 5), step)]
    while len(selected) < 5:
        selected.append(question)

    return selected[:5]

# --------------------------------------------
# 🚀 Main paraphrase generation
# --------------------------------------------
def generate_paraphrases():
    cfg = CONFIG
    input_path, output_path = cfg["input_path"], cfg["output_path"]

    print(f"🔹 Loading base questions from: {input_path}")
    lines = [json.loads(l) for l in open(input_path, encoding="utf-8")]
    total = len(lines)
    if cfg["offset"]:
        lines = lines[cfg["offset"]:]
    if cfg["limit"]:
        lines = lines[:cfg["limit"]]
    random.seed(42)  # for reproducibility
    random.shuffle(lines)
    print(f"✅ Selected {len(lines)} examples (from {total})")

    # Determine device
    if cfg["device_preference"].lower() == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = cfg["device_preference"]
    print(f"💻 Using device: {device.upper()}")

    # Load models
    print(f"🧠 Loading paraphrase model: {cfg['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"]).to(device)
    model.eval()

    print(f"🔎 Loading embedding model: {cfg['embedding_model']}")
    embedder = SentenceTransformer(cfg["embedding_model"], device=device)

    # Paraphrase loop
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for row in tqdm(lines, desc="Generating paraphrases"):
            qid = row["id"]
            question = row["question"]
            answer = row["answer"]

            # 1️⃣ Always keep degree=1 identical
            degree_1 = question.strip()

            # 2️⃣ Generate many candidates
            candidates = paraphrase_candidates(
                model, tokenizer, question, device, cfg["max_new_tokens"], cfg["num_candidates"]
            )

            # 3️⃣ Select 4 progressively different but semantically close paraphrases
            selected = select_progressive_paraphrases(question, candidates, embedder)[1:5]

            # 4️⃣ Write all 5 degrees (1 = original, 2–5 = selected)
            records = [(1, degree_1)] + [(i + 2, s) for i, s in enumerate(selected)]

            for degree, paraphrased in records:
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
# 🏁 Entry point
# --------------------------------------------
if __name__ == "__main__":
    cfg = CONFIG
    generate_initial_dataset(cfg)
    generate_paraphrases()
