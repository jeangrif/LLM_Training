import optuna
import subprocess
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

STUDY_NAME = "rag_param_retrieval_only_augmented_question_without_latency_penalty_remove_useless_alphav2"
STORAGE = "sqlite:///optuna_study.db"
N_TOP = 5
MAX_QUESTIONS = 10000
OUTPUT_ROOT = Path("outputs/validation_10k")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def format_value(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return str(v).lower()
    return v

def find_summary_in_dir(output_dir: Path) -> Path:
    path = output_dir / "summarize_experiment.json"
    if not path.exists():
        raise FileNotFoundError(f"Pas de summarize_experiment.json dans {output_dir}")
    return path

def run_trial(params: dict, tag: str) -> dict:
    overrides = [f"{k}={format_value(v)}" for k, v in params.items()]
    overrides.append(f"modules.run_rag.max_questions={MAX_QUESTIONS}")
    overrides.append(f"output_dir={OUTPUT_ROOT / tag}")

    print(f"\n🚀 Exécution {tag} : {' '.join(overrides)}")
    subprocess.run(["python", "-m", "src.main"] + overrides, check=True)

    summary_path = find_summary_in_dir(OUTPUT_ROOT / tag)
    print(f"✅ Résumé trouvé : {summary_path}")

    with open(summary_path, encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("evaluation_metrics", {}) or {}
    latency = data.get("generation", {}).get("mean", None)

    return {
        "tag": tag,
        **params,
        **metrics,
        "latency_mean_s": latency,
    }

def main():
    print("📊 Chargement de l'étude Optuna...")
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:N_TOP]

    print(f"🏆 {len(top_trials)} meilleurs essais sélectionnés pour validation sur {MAX_QUESTIONS} questions")

    results = []
    for i, trial in enumerate(top_trials, 1):
        tag = f"trial_{trial.number}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n==== 🧪 Validation {i}/{N_TOP} — trial #{trial.number} (score optuna={trial.value:.4f}) ====")
        try:
            res = run_trial(trial.params, tag)
            res["optuna_value"] = trial.value
            results.append(res)
        except subprocess.CalledProcessError:
            print(f"❌ Échec du run pour trial {trial.number}")
        except Exception as e:
            print(f"⚠️ Erreur inattendue sur trial {trial.number} : {e}")

    if results:
        df = pd.DataFrame(results)
        csv_path = OUTPUT_ROOT / f"validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Résumé complet sauvegardé : {csv_path}")
    else:
        print("⚠️ Aucun résultat exploitable obtenu.")

if __name__ == "__main__":
    main()
