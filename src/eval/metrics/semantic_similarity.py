# src/eval/metrics.yaml/semantic_similarity.py
from sentence_transformers import util

def semantic_similarity(pred, gold, model):
    emb1 = model.encode(pred, convert_to_tensor=True, show_progress_bar=False)
    emb2 = model.encode(gold, convert_to_tensor=True, show_progress_bar=False)
    return util.cos_sim(emb1, emb2).item()
