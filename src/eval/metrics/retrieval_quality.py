from sentence_transformers import util
import torch

def retrieval_quality(query, retrieved_docs, relevant_docs, model, k=5, relevance_threshold=0.7):
    """
    Évalue la qualité du retrieval d'un RAG via trois critères :
    - recall@k : proportion de documents pertinents retrouvés
    - mmr_diversity : redondance entre les documents
    - embedding_coverage : proximité sémantique moyenne avec la requête
    """

    if not retrieved_docs:
        return {"recall@k": 0.0, "mmr_diversity": 0.0, "embedding_coverage": 0.0}

    # Encodage
    emb_query = model.encode(query, convert_to_tensor=True, show_progress_bar=False)
    emb_retrieved = model.encode(retrieved_docs, convert_to_tensor=True, show_progress_bar=False)

    # Si ground truth dispo
    if relevant_docs:
        emb_relevant = model.encode(relevant_docs, convert_to_tensor=True, show_progress_bar=False)
    else:
        emb_relevant = None

    # -------------------------
    # 1. Recall@k
    # -------------------------
    recall_k = 0.0
    if emb_relevant is not None:
        sims = util.cos_sim(emb_retrieved[:k], emb_relevant)
        hits = (sims > relevance_threshold).any(dim=1)
        recall_k = hits.float().mean().item()

    # -------------------------
    # 2. MMR Diversity
    # -------------------------
    mmr_div = 0.0
    if len(retrieved_docs) > 1:
        sims = util.cos_sim(emb_retrieved, emb_retrieved)
        sims.fill_diagonal_(0.0)
        avg_sim = sims.mean().item()
        mmr_div = 1 - avg_sim  # plus c'est haut, plus c'est diversifié

    # -------------------------
    # 3. Embedding Coverage
    # -------------------------
    coverage = util.cos_sim(emb_query, emb_retrieved).mean().item()

    return {
        "recall@k": recall_k,
        "mmr_diversity": mmr_div,
        "embedding_coverage": coverage,
    }
