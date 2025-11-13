import optuna
import subprocess
from pathlib import Path
import json

# ============================
#        GLOBAL CONFIG
# ============================
PREV_STUDY_NAME = "rag_param_retrieval_only_initial_question_precision_penalized"
STORAGE = "sqlite:///optuna_study.db"
N_TOP_CONFIGS = 5
N_TRIALS = 50  # total trials across ALL base configs in a single study

# LLM HP to explore
TEMP_RANGE = (0.0, 1.0)
TOPP_RANGE = (0.6, 1.0)

# Metric weights (exact_match prioritized)
W_EM = 0.80
W_FAITH = 0.15
W_ROB = 0.05

LATENCY_MODE = "penalized"
ALPHA_PER_SEC = 0.00
LATENCY_BUDGET_MS = 1500
LATENCY_STRATEGY = "mean"


# ============================
#         HELPERS
# ============================
def find_latest_summary():
    outputs_root = Path("outputs")
    summaries = list(outputs_root.rglob("summarize_experiment.json"))
    if not summaries:
        raise FileNotFoundError("⚠️ Aucun fichier summarize_experiment.json trouvé.")
    return max(summaries, key=lambda p: p.stat().st_mtime)


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
    return total_sec * 1000.0


def _pick_quality(metrics: dict) -> float:
    """Objectif LLM : exact_match prioritaire"""
    em = metrics.get("exact_match", metrics.get("lexical_overlap", 0.0))
    f = metrics.get("faithfulness", 0.0)
    rb = metrics.get("robustness", 0.0)
    em, f, rb = float(em or 0.0), float(f or 0.0), float(rb or 0.0)
    if any([em, f, rb]):
        return W_EM * em + W_FAITH * f + W_ROB * rb

    # fallback on retrieval metrics if generation metrics are absent
    rq = metrics.get("retriever_quality")
    if isinstance(rq, dict):
        ndcg = float(rq.get("nDCG", 0))
        recall = max(float(rq.get("recall@5", 0)), float(rq.get("recall", 0)))
        precision = max(float(rq.get("precision@5", 0)), float(rq.get("precision", 0)))
        return 0.6 * ndcg + 0.25 * precision + 0.15 * recall

    raise ValueError("⚠️ Aucune métrique utilisable trouvée.")


def run_pipeline_with_overrides(overrides):
    cmd = ["python", "-m", "src.main"] + overrides
    print(f"🚀 Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    summary_path = find_latest_summary()
    with open(summary_path, encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("evaluation_metrics", {}) or {}
    quality = _pick_quality(metrics)
    latency_ms = _compute_latency_ms(data)

    if LATENCY_MODE == "budget":
        if latency_ms > LATENCY_BUDGET_MS:
            raise optuna.TrialPruned(f"⏱️ {latency_ms:.0f}ms > {LATENCY_BUDGET_MS}ms")
        return quality
    return quality - ALPHA_PER_SEC * (latency_ms / 1000.0)


# ============================
#     RETRIEVE TOP CONFIGS
# ============================
def load_top_configs(n=5):
    study = optuna.load_study(study_name=PREV_STUDY_NAME, storage=STORAGE)
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trials.sort(key=lambda t: t.value if t.value is not None else float("-inf"), reverse=True)
    best = []
    seen = set()
    for t in trials:
        params = t.params
        key = tuple(sorted(params.items()))
        if key not in seen:
            seen.add(key)
            best.append(params)
        if len(best) >= n:
            break
    print(f"🔹 {len(best)} meilleures configs récupérées depuis {PREV_STUDY_NAME}")
    return best


# ============================
#        SINGLE OBJECTIVE
# ============================
def make_objective_all_configs(top_configs):
    """
    Single objective that:
    - chooses one of the top retrieval/rerank configs as a categorical option
    - tunes only LLM params (temperature, top_p)
    """
    def objective(trial: optuna.Trial):
        # Choose which frozen base config to use this trial
        base_idx = trial.suggest_int("base_config_id", 0, len(top_configs) - 1)
        base_params = top_configs[base_idx]

        # LLM params
        temperature = trial.suggest_float("model.temperature", *TEMP_RANGE)
        top_p = trial.suggest_float("model.top_p", *TOPP_RANGE)

        # Build overrides from the chosen base config (freeze retriever/reranker)
        overrides = [
            f"modules.run_rag.retrieval_type={base_params['modules.run_rag.retrieval_type']}",
            f"modules.run_rag.top_k={base_params['modules.run_rag.top_k']}",
            f"embed.embedding_model={base_params['embed.embedding_model']}",
            f"embed.chunk_size={base_params['embed.chunk_size']}",
            f"modules.run_rag.use_rerank={base_params['modules.run_rag.use_rerank']}",
            "modules.run_rag.do_generation=true",
            f"model.temperature={temperature}",
            f"model.top_p={top_p}",
        ]

        # Optional: rerank top_k if available under either key
        rerank_k = base_params.get("rerank.top_k", base_params.get("modules.run_rag.top_k_rerank"))
        if rerank_k is not None:
            overrides.append(f"rerank.top_k={rerank_k}")

        # Optional: alpha if hybrid
        if base_params.get("modules.run_rag.retrieval_type") == "hybrid":
            alpha = base_params.get("modules.run_rag.alpha")
            if alpha is not None:
                overrides.append(f"modules.run_rag.alpha={alpha}")

        return run_pipeline_with_overrides(overrides)

    return objective


# ============================
#            MAIN
# ============================
if __name__ == "__main__":
    top_configs = load_top_configs(N_TOP_CONFIGS)

    # One unified study for ALL configs + LLM space
    study = optuna.create_study(
        study_name="rag_stage2_llm_frozen_retrievalv2_all",
        storage=STORAGE,
        direction="maximize",
        load_if_exists=True,
    )
    print("\n=== 🚀 Lancement LLM Opti sur un SEUL study (configs top-N en catégoriel) ===")
    study.optimize(make_objective_all_configs(top_configs), n_trials=N_TRIALS)

    print("\n✅ Stage-2 terminé (single study)")
    print("🏆 Meilleurs paramètres :", study.best_params)
    print("📈 Score :", study.best_value)
