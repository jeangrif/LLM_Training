# -*- coding: utf-8 -*-
"""
Generate answers (RAG vs no-RAG) for the evaluation dataset.
Output is a JSONL file ready for metric evaluation.
"""
from pathlib import Path
import os, sys, json
from tqdm import tqdm
from dotenv import load_dotenv

# Standard header for PyCharm
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.rag.pipeline import RagPipeline
from src.utils.jsonl_helper import load_jsonl

# ------------------------------------------------------
# Main
# ------------------------------------------------------
def main():
    load_dotenv()

    input_path = Path(os.getenv("EVAL_INPUT", "data/eval/augmented_questions.jsonl"))
    output_path_rag = Path(os.getenv("EVAL_RESULTS_RAG", "data/eval/results_rag.jsonl"))
    output_path_no_rag = Path(os.getenv("EVAL_RESULTS_NO_RAG", "data/eval/results_no_rag.jsonl"))
    output_path_rag.parent.mkdir(parents=True, exist_ok=True)

    # 🔹 Load base questions
    questions = load_jsonl(input_path)
    print(f"✅ Loaded {len(questions)} questions from {input_path}")

    # 🔹 Init RAG pipeline once
    rag = RagPipeline(top_k=3)

    # ------------------------------------------------------
    # Generate predictions with and without RAG
    # ------------------------------------------------------
    with open(output_path_rag, "w", encoding="utf-8") as fr, \
         open(output_path_no_rag, "w", encoding="utf-8") as fn:

        for row in tqdm(questions, desc="Generating answers"):
            qid = row["orig_id"]
            degree = row["degree"]
            question = row["question"]
            answer = row["answer"]

            try:
                rag.generator.model.reset()  # reset memory
                response_rag = rag.answer(question)
                contexts = [c["text"] for c in rag.retriever.retrieve(question, top_k=3)]
            except Exception as e:
                response_rag = f"[Error RAG: {e}]"
                contexts = []

            # --- Without RAG ---
            try:
                rag.generator.model.reset()  # reset memory again
                response_no_rag = rag.generator.model.generate(question)
            except Exception as e:
                response_no_rag = f"[Error noRAG: {e}]"

            # --- Save both ---
            record_rag = {
                "orig_id": qid,
                "degree": degree,
                "question": question,
                "answer": answer,
                "pred": response_rag,
                "contexts": contexts
            }
            record_no_rag = {
                "orig_id": qid,
                "degree": degree,
                "question": question,
                "answer": answer,
                "pred": response_no_rag,
                "contexts": []
            }

            fr.write(json.dumps(record_rag, ensure_ascii=False) + "\n")
            fn.write(json.dumps(record_no_rag, ensure_ascii=False) + "\n")

    rag.generator.close()
    print(f"\n✅ Saved RAG results to {output_path_rag}")
    print(f"✅ Saved no-RAG results to {output_path_no_rag}")


if __name__ == "__main__":
    main()
