import json
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# === Config ===
FILE_PATH = Path("/Users/jeangrifnee/PycharmProjects/LLMTraining/results/2025-10-22/20-17-18__rag_dense_top3_temp0.70/eval_summary.jsonl")  # ton fichier d'éval
STEP_SIZES = [10, 50, 100, 150, 200, 300, 400, 500]
METRICS = ["exact_match", "semantic_sim", "faithfulness", "robustness"]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def compute_convergence(data, metrics, step_sizes):
    curves = {m: {"mean": [], "std": []} for m in metrics}
    n = len(data)

    for step in step_sizes:
        subset = data[:min(step, n)]
        for m in metrics:
            vals = [r[m] for r in subset if m in r and isinstance(r[m], (int, float))]
            if vals:
                curves[m]["mean"].append(np.mean(vals))
                curves[m]["std"].append(np.std(vals))
            else:
                curves[m]["mean"].append(np.nan)
                curves[m]["std"].append(np.nan)

    return curves


def detect_stabilization(values, steps, threshold=0.01, window=3):
    """
    Renvoie le premier step où la variation moyenne < threshold pendant `window` paliers.
    """
    if len(values) < window + 1:
        return None

    for i in range(len(values) - window):
        diffs = np.abs(np.diff(values[i:i + window + 1]))
        rel_diffs = diffs / np.maximum(values[i:i + window], 1e-6)
        if np.all(rel_diffs < threshold):
            return steps[i + window]
    return None


def plot_convergence(curves, step_sizes):
    fig = go.Figure()

    for metric, stats in curves.items():
        mean_vals = np.array(stats["mean"])
        std_vals = np.array(stats["std"])

        # courbe principale
        fig.add_trace(go.Scatter(
            x=step_sizes, y=mean_vals,
            mode="lines+markers",
            name=metric
        ))

        # bandes ± écart-type
        fig.add_trace(go.Scatter(
            x=list(step_sizes) + list(step_sizes[::-1]),
            y=list(mean_vals + std_vals) + list((mean_vals - std_vals)[::-1]),
            fill="toself",
            fillcolor="rgba(0,0,0,0.05)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False
        ))

        # point de stabilisation
        stable = detect_stabilization(mean_vals, step_sizes)
        if stable:
            idx = step_sizes.index(stable)
            fig.add_trace(go.Scatter(
                x=[stable],
                y=[mean_vals[idx]],
                mode="markers+text",
                marker=dict(color="green", size=10, symbol="diamond"),
                text=[f"Stabilized @ {stable}"],
                textposition="top center",
                name=f"{metric} stabilized"
            ))

    fig.update_layout(
        title="Évolution des scores moyens et stabilité des métriques",
        xaxis_title="Nombre de questions évaluées",
        yaxis_title="Score moyen",
        template="plotly_white",
        width=950,
        height=550,
    )

    fig.show()


if __name__ == "__main__":
    data = load_jsonl(FILE_PATH)
    print(f"✅ Loaded {len(data)} rows from {FILE_PATH}")

    curves = compute_convergence(data, METRICS, STEP_SIZES)
    plot_convergence(curves, STEP_SIZES)
