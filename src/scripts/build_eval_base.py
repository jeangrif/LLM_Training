from pathlib import Path
import os, sys

# forcer le cwd sur la racine du projet
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.rag.setting import settings

def main():
    raw_path = Path(settings.RAW_DIR) / f"{settings.HF_DATASET_NAME}_{settings.HF_DATASET_SPLIT}.parquet"
    out_path = Path("data/eval/base_questions.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"🔹 Loading dataset: {raw_path}")
    df = pd.read_parquet(raw_path)

    # Vérifie que les colonnes existent
    required_cols = {"id", "question", "answers"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing expected columns: {required_cols - set(df.columns)}")

    print(f"✅ Loaded {len(df)} rows")

    # (Optionnel) Prendre un sous-échantillon
    limit = int(settings.EVAL_LIMIT) if hasattr(settings, "EVAL_LIMIT") else None
    if limit:
        df = df.sample(n=limit, random_state=42)
        print(f"🔹 Sampling {limit} examples")

    with open(out_path, "w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            answers = row["answers"]

            # Récupère le texte de réponse principal
            if isinstance(answers, dict) and "text" in answers and len(answers["text"]) > 0:
                answer = answers["text"][0]
            else:
                continue

            record = {
                "id": row["id"],
                "question": row["question"],
                "answer": answer
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Saved base eval set to {out_path}")


if __name__ == "__main__":
    main()