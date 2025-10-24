# src/eval/metrics.yaml/robustness.py
from sentence_transformers import util
import numpy as np

def robustness_by_degree(group, model):
    embs = [model.encode(row["pred"], convert_to_tensor=True, show_progress_bar=False) for row in group]
    sims = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            sims.append(util.cos_sim(embs[i], embs[j]).item())
    return float(np.mean(sims))

