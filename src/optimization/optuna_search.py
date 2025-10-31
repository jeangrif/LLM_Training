import optuna
import subprocess
from pathlib import Path
import json

# --- Réglages latence (inchangés par rapport à la version qui marche) ---
LATENCY_MODE = "penalized"   # "penalized" | "budget"
ALPHA_PER_SEC = 0.00        # pénalité par seconde si "penalized"
LATENCY_BUDGET_MS = 1500     # budget si "budget" (ms)
LATENCY_STRATEGY = "mean"    # "mean" | "mean_plus_2std"
RETRIEVER_OPT_MODE = True
SEEN_EFFECTIVE_CONFIGS = set()

# --- Nouveaux espaces de recherche (à adapter à ton setup) ---
EMBEDDING_CHOICES = [
    # ← mets ici tes 2 modèles d'embeddings (exemples ci-dessous)
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2"
]
CHUNK_SIZES = [256, 384, 512, 768, 1024]  # ← plusieurs tailles de chunking

def find_latest_summary():
    outputs_root = Path("outputs")
    summaries = list(outputs_root.rglob("summarize_experiment.json"))
    if not summaries:
        raise FileNotFoundError("⚠️ Aucun fichier summarize_experiment.json trouvé dans outputs/")
    latest = max(summaries, key=lambda p: p.stat().st_mtime)
    return latest

def _compute_latency_ms(summary: dict) -> float:
    def part(stage: str) -> float:
        d = summary.get(stage, {})
        if not isinstance(d, dict):
            return 0.0
        mean = float(d.get("mean") or 0.0)
        std = float(d.get("std") or 0.0)
        if LATENCY_STRATEGY == "mean_plus_2std":
            return max(mean + 2 * std, 0.0)
        return max(mean, 0.0)

    total_sec = part("retrieval") + part("rerank") + part("generation")
    return total_sec * 1000.0  # ms

def _pick_quality(metrics: dict) -> float:
    """
    Si des métriques de retrieval sont présentes (retriever_quality dict),
    on calcule un score combiné. Sinon on retombe sur les métriques génération.
    """
    rq = metrics.get("retriever_quality")
    if isinstance(rq, dict):
        def g(key, default=0.0):
            v = rq.get(key, default)
            return float(v) if isinstance(v, (int, float)) else 0.0

        # pondérations simples et robustes
        ndcg      = g("nDCG")
        recall    = max(g("recall@5"), g("recall"))
        precision = max(g("precision@5"), g("precision"))

        # 👇 formule objective pour le retrieval-only
        # - nDCG capte le ranking (principal)
        # - recall@k capte la couverture
        # - precision@k évite d'inonder de faux positifs
        return 0.7 * ndcg + 0.3 * recall

    # --- fallback génération (ton comportement actuel) ---
    for k in ["faithfulness", "exact_match", "semantic_similarity", "lexical_overlap", "semantic_sim"]:
        v = metrics.get(k)
        if v is not None:
            return float(v)

    raise ValueError("⚠️ Aucune métrique utilisable trouvée (retrieval ou génération).")


def run_pipeline_with_overrides(overrides):
    cmd = ["python", "-m", "src.main"] + overrides
    print(f"🚀 Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # NE CHANGE PAS la logique d'accès au fichier : on prend le dernier résumé
    summary_path = find_latest_summary()
    print(f"✅ Résumé trouvé : {summary_path}")

    with open(summary_path, encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("evaluation_metrics", {}) or {}
    quality = _pick_quality(metrics)
    latency_ms = _compute_latency_ms(data)

    if LATENCY_MODE == "budget":
        if latency_ms > LATENCY_BUDGET_MS:
            raise optuna.TrialPruned(f"⏱️ Latence {latency_ms:.0f}ms > budget {LATENCY_BUDGET_MS}ms")
        final = quality
        print(f"🎯 Qualité: {quality:.4f} | ⏱️ {latency_ms:.0f} ms | ✅ sous budget -> score={final:.4f}")
    else:
        final = quality - ALPHA_PER_SEC * (latency_ms / 1000.0)
        print(f"🎯 Qualité: {quality:.4f} | ⏱️ {latency_ms:.0f} ms | 📉 score pénalisé={final:.4f} (alpha={ALPHA_PER_SEC}/s)")

    return final

def objective(trial):
    retrieval_type = trial.suggest_categorical(
        "modules.run_rag.retrieval_type", ["dense", "hybrid", "sparse"]
    )

    top_k = trial.suggest_categorical(
        "modules.run_rag.top_k", [2, 4, 8, 16, 24]
    )

    # ✅ On définit toujours le même espace pour éviter le "dynamic value space"
    top_k_rerank = trial.suggest_categorical(
        "modules.run_rag.top_k_rerank", [2, 4, 8, 16, 24, None]
    )

    # ✅ On invalide ensuite les valeurs incohérentes selon top_k
    if top_k_rerank is not None and top_k_rerank > top_k:
        # soit on corrige :
        top_k_rerank = top_k
        # soit on prune (si tu veux que le trial soit ignoré) :
        # raise optuna.TrialPruned("Invalid top_k_rerank > top_k")

    embedding_model = trial.suggest_categorical(
        "embed.embedding_model", EMBEDDING_CHOICES
    )
    chunk_size = trial.suggest_categorical(
        "embed.chunk_size", CHUNK_SIZES
    )
    use_rerank = trial.suggest_categorical(
        "modules.run_rag.use_rerank", [True, False]
    )
    if use_rerank and top_k_rerank is None:
        raise optuna.TrialPruned("Invalid combination: use_rerank=True but top_k_rerank=None")
    if top_k_rerank is not None and top_k_rerank > top_k:
        top_k_rerank = top_k

    alpha = None
    if retrieval_type == "hybrid":
        alpha = trial.suggest_float("modules.run_rag.alpha", 0.3, 0.8)

    # --- clé effective pour déduplication ---
    key = (
        retrieval_type,
        top_k,
        top_k_rerank,
        embedding_model,
        chunk_size,
        use_rerank,
        alpha if retrieval_type == "hybrid" else None,
    )
    if key in SEEN_EFFECTIVE_CONFIGS:
        raise optuna.TrialPruned("Duplicate effective configuration.")
    SEEN_EFFECTIVE_CONFIGS.add(key)

    overrides = [
        f"modules.run_rag.retrieval_type={retrieval_type}",
        f"modules.run_rag.top_k={top_k}",
        f"rerank.top_k={top_k_rerank}",
        f"embed.embedding_model={embedding_model}",
        f"embed.chunk_size={chunk_size}",
        f"modules.run_rag.use_rerank={use_rerank}",
    ]
    if retrieval_type == "hybrid":
        overrides.append(f"modules.run_rag.alpha={alpha}")
    if RETRIEVER_OPT_MODE:
        overrides.append("modules.run_rag.do_generation=false")

    return run_pipeline_with_overrides(overrides)





if __name__ == "__main__":
    # Étude Optuna persistée en SQLite
    study = optuna.create_study(
        study_name="rag_param_retrieval_only_augmented_question_without_latency_penalty_remove_useless_alphav2",
        storage="sqlite:///optuna_study.db",
        direction="maximize",
        load_if_exists=True,
    )
    # Augmente un peu les essais pour explorer les nouvelles dimensions
    study.optimize(objective, n_trials=60)

    print("\n✅ Optimisation terminée !")
    print("🏆 Meilleurs paramètres :", study.best_params)
    print("📈 Score obtenu :", study.best_value)
