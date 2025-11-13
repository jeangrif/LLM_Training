DEFAULT_GUARDRAIL_PRESETS = {
    "dense":  {"min_abs": 0.15, "good_abs": 0.35, "min_margin_iqr": 0.5, "soft": True},
    "sparse": {"min_abs": 5.0,  "good_abs": 15.0, "min_margin_iqr": 0.5, "soft": True},
    "hybrid": {"min_abs": 0.15, "good_abs": 0.35, "min_margin_iqr": 0.5, "soft": True},
}

def _rank_normalize(values: list[float]) -> list[float]:
    """Transforme une liste de scores (éventuellement avec None) en rang-normalisé dans [0,1], 1 = meilleur."""
    if values is None:
        return []
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    # Remplace None par -inf pour les classer tout en bas
    safe_vals = [float("-inf") if v is None else float(v) for v in values]
    # argsort décroissant
    idx = sorted(range(n), key=lambda i: safe_vals[i], reverse=True)
    rank_norm = [0.0] * n
    denom = n - 1
    for rank, i in enumerate(idx, start=1):
        # 1 - (rank-1)/(n-1) -> meilleur = 1.0, pire = 0.0
        rank_norm[i] = 1.0 - (rank - 1) / denom
    return rank_norm


def attach_unified_scores(
    docs: list[dict],
    retrieval_type: str,
    alpha: float = 0.5,   # poids hybrid (dense vs sparse)
    w_rerank: float = 0.7, # poids rerank vs base
    guardrails_cfg: dict | None = None,
) -> list[float]:
    """
    Prend une liste de docs contenant éventuellement :
      - 'dense_score', 'sparse_score', 'fused_score', 'rerank_score'
    Retourne une liste 'scores' alignée avec docs, normalisée dans [0,1].
    Applique un garde-fou adaptatif (paramètres fixés en dur) pour couper/atténuer quand c'est trop faible.
    """
    n = len(docs)
    if n == 0:
        return []

    # -------------------------------
    # Garde-fou : paramètres EN DUR
    # -------------------------------
    presets = guardrails_cfg or {}
    gr = (
            presets.get(retrieval_type)
            or presets.get("default")
            or DEFAULT_GUARDRAIL_PRESETS[retrieval_type]
    )


    # Colonnes présentes
    have_dense  = any("dense_score"  in d for d in docs)
    have_sparse = any("sparse_score" in d for d in docs)
    have_fused  = any("fused_score"  in d for d in docs)
    have_rerank = any("rerank_score" in d for d in docs)

    dense_vals  = [d.get("dense_score")  for d in docs] if have_dense  else None
    sparse_vals = [d.get("sparse_score") for d in docs] if have_sparse else None
    fused_vals  = [d.get("fused_score")  for d in docs] if have_fused  else None
    rerank_vals = [d.get("rerank_score") for d in docs] if have_rerank else None

    # Rank-normalisation
    dense_norm  = _rank_normalize(dense_vals)  if dense_vals  is not None else None
    sparse_norm = _rank_normalize(sparse_vals) if sparse_vals is not None else None
    fused_norm  = _rank_normalize(fused_vals)  if fused_vals  is not None else None
    rerank_norm = _rank_normalize(rerank_vals) if rerank_vals is not None else None

    # Base_norm (sans rerank)
    def compute_base_norm() -> list[float]:
        # hybrid : on préfère fused s'il existe
        if retrieval_type == "hybrid":
            if fused_norm is not None:
                return fused_norm
            if dense_norm is not None and sparse_norm is not None:
                return [alpha * dn + (1 - alpha) * sn for dn, sn in zip(dense_norm, sparse_norm)]
            if dense_norm is not None:
                return dense_norm
            if sparse_norm is not None:
                return sparse_norm
        # dense / sparse
        if retrieval_type == "dense" and dense_norm is not None:
            return dense_norm
        if retrieval_type == "sparse" and sparse_norm is not None:
            return sparse_norm
        # fallback : première norme dispo
        for col in (fused_norm, dense_norm, sparse_norm):
            if col is not None:
                return col
        # si aucune colonne présente
        return [0.0] * n

    base_norm = compute_base_norm()

    # Sans rerank → base_norm ; avec rerank → mélange
    if rerank_norm is None:
        final_scores = [float(x) for x in base_norm]
    else:
        final_scores = [float(w_rerank * r + (1.0 - w_rerank) * b) for r, b in zip(rerank_norm, base_norm)]

    # -------------------------
    # Garde-fou adaptatif (dur)
    # -------------------------
    def _percentile(seq: list[float], p: float) -> float:
        s = sorted(seq)
        if not s:
            return float("-inf")
        k = (len(s) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(s) - 1)
        return s[f] + (s[c] - s[f]) * (k - f)

    # colonne brute "opérative" alignée avec la combinaison réellement utilisée
    def _operative_raw() -> list[float]:
        # base raw selon retrieval_type
        if retrieval_type == "hybrid":
            if fused_vals is not None:
                base_raw = fused_vals
            elif dense_vals is not None and sparse_vals is not None:
                base_raw = [
                    alpha * (d if d is not None else float("-inf")) +
                    (1 - alpha) * (s if s is not None else float("-inf"))
                    for d, s in zip(dense_vals, sparse_vals)
                ]
            elif dense_vals is not None:
                base_raw = dense_vals
            elif sparse_vals is not None:
                base_raw = sparse_vals
            else:
                base_raw = [float("-inf")] * n
        elif retrieval_type == "dense" and dense_vals is not None:
            base_raw = dense_vals
        elif retrieval_type == "sparse" and sparse_vals is not None:
            base_raw = sparse_vals
        else:
            base_raw = fused_vals or dense_vals or sparse_vals or [float("-inf")] * n

        # mélange avec rerank si présent (mêmes poids que le final)
        if rerank_vals is not None:
            return [
                w_rerank * (r if r is not None else float("-inf")) +
                (1.0 - w_rerank) * (b if b is not None else float("-inf"))
                for r, b in zip(rerank_vals, base_raw)
            ]
        return [(v if v is not None else float("-inf")) for v in base_raw]

    oper_raw = _operative_raw()
    clean = [v for v in oper_raw if v is not None and v != float("-inf")]
    if not clean:
        return [0.0] * n

    top = max(clean)
    sc = sorted(clean, reverse=True)
    second = sc[1] if len(sc) >= 2 else float("-inf")
    q25 = _percentile(clean, 25.0)
    q75 = _percentile(clean, 75.0)
    iqr = max(1e-9, q75 - q25)

    # Gate 1 : qualité absolue (rampe min_abs -> good_abs)
    g_abs = 1.0
    if gr["min_abs"] is not None:
        min_abs = float(gr["min_abs"])
        good_abs = gr.get("good_abs", None)
        if good_abs is None:
            g_abs = 0.0 if top < min_abs else 1.0
        else:
            good_abs = float(good_abs)
            if good_abs <= min_abs:
                good_abs = min_abs + 1e-6
            g_abs = max(0.0, min(1.0, (top - min_abs) / (good_abs - min_abs)))

    # Gate 2 : séparation relative (marge top-2 vs IQR)
    margin = max(0.0, top - (second if second != float("-inf") else q75))
    k = float(gr["min_margin_iqr"])
    g_sep = max(0.0, min(1.0, (margin - k * iqr) / max(1e-9, iqr)))

    # Combinaison des deux gardes (soft = min, hard = AND binaire)
    if gr.get("soft", True):
        g = min(g_abs, g_sep)
    else:
        g = 1.0 if (g_abs >= 1.0 and g_sep >= 1.0) else 0.0

    if g <= 0.0:
        return [0.0] * n

    return [s * g for s in final_scores]
