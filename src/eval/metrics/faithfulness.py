# src/eval/metrics.yaml/faithfulness.py
from sentence_transformers import util

def faithfulness_score(pred, contexts, model):
    if not contexts:
        return 0.0
    emb_pred = model.encode(pred, convert_to_tensor=True, show_progress_bar=False)
    scores = [
        util.cos_sim(emb_pred, model.encode(ctx, convert_to_tensor=True, show_progress_bar=False)).item()
        for ctx in contexts
    ]
    return max(scores)

