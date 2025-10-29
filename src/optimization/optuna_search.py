import optuna
import subprocess
from pathlib import Path
import json

def find_latest_summary():
    outputs_root = Path("outputs")
    summaries = list(outputs_root.rglob("summarize_experiment.json"))
    if not summaries:
        raise FileNotFoundError("⚠️ Aucun fichier summarize_experiment.json trouvé dans outputs/")
    latest = max(summaries, key=lambda p: p.stat().st_mtime)
    return latest

def run_pipeline_with_overrides(overrides):
    cmd = ["python", "-m", "src.main"] + overrides
    print(f"🚀 Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Trouve automatiquement le résumé le plus récent
    summary_path = find_latest_summary()
    print(f"✅ Résumé trouvé : {summary_path}")

    with open(summary_path) as f:
        data = json.load(f)

    metrics = data.get("evaluation_metrics", {})
    score = (
        metrics.get("faithfulness")
        or metrics.get("semantic_sim")
        or metrics.get("exact_match")
    )

    if score is None:
        raise ValueError(f"⚠️ Aucune métrique utilisable trouvée dans {summary_path}")

    print(f"🎯 Score utilisé pour Optuna : {score:.4f}")
    return score


def objective(trial):
    retrieval_type = trial.suggest_categorical(
        "modules.run_rag.retrieval_type", ["dense", "hybrid", "sparse"]
    )
    top_k = trial.suggest_categorical(
        "modules.run_rag.top_k", [2, 4, 8, 16]
    )

    overrides = [
        f"modules.run_rag.retrieval_type={retrieval_type}",
        f"modules.run_rag.top_k={top_k}",
    ]

    return run_pipeline_with_overrides(overrides)

if __name__ == "__main__":
    # Simple étude Optuna en mémoire (pas de stockage)
    study = optuna.create_study(
        study_name="rag_param_search",
        storage="sqlite:///optuna_study.db",
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=6)

    print("\n✅ Optimisation terminée !")
    print("🏆 Meilleurs paramètres :", study.best_params)
    print("📈 Score obtenu :", study.best_value)
