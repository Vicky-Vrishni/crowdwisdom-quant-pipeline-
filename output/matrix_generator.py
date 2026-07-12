import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUTPUT_DIR, SIMULATION_PERMUTATIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("matrix_generator")

WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DEFAULT_BASELINE_PERMUTATION = SIMULATION_PERMUTATIONS[0]


def build_best_permutation_matrix(predictions: pd.DataFrame) -> pd.DataFrame:
    cell_scores = predictions.groupby(
        ["weekday", "hour_of_day", "simulation_permutation"], as_index=False
    )["predicted_pnl"].mean()

    best = cell_scores.loc[cell_scores.groupby(["weekday", "hour_of_day"])["predicted_pnl"].idxmax()]
    return best.sort_values(["weekday", "hour_of_day"])


def plot_heatmap(best_matrix: pd.DataFrame, save_path):
    perms = sorted(best_matrix["simulation_permutation"].unique())
    perm_idx = {p: i for i, p in enumerate(perms)}

    grid_score = np.full((7, 24), np.nan)
    grid_perm = np.full((7, 24), -1, dtype=int)

    for _, row in best_matrix.iterrows():
        w, h = int(row["weekday"]), int(row["hour_of_day"])
        grid_score[w, h] = row["predicted_pnl"]
        grid_perm[w, h] = perm_idx[row["simulation_permutation"]]

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(grid_score, aspect="auto", cmap="RdYlGn")

    for w in range(7):
        for h in range(24):
            if grid_perm[w, h] >= 0:
                label = perms[grid_perm[w, h]].split("_")[0]  # short code e.g. "A"
                ax.text(h, w, label, ha="center", va="center", fontsize=7, color="black")

    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.set_yticks(range(7))
    ax.set_yticklabels(WEEKDAY_LABELS)
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Weekday")
    ax.set_title("Best Simulation Permutation by Hour x Weekday (out-of-sample predicted P&L)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Predicted mean P&L")

    legend_text = "  |  ".join(f"{p.split('_')[0]} = {p}" for p in perms)
    fig.text(0.5, -0.02, legend_text, ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved heatmap to %s", save_path)


def build_equity_curves(predictions: pd.DataFrame, baseline_permutation: str = DEFAULT_BASELINE_PERMUTATION):
    best_by_cell = build_best_permutation_matrix(predictions)[
        ["weekday", "hour_of_day", "simulation_permutation"]
    ].rename(columns={"simulation_permutation": "recommended_permutation"})

    merged = predictions.merge(best_by_cell, on=["weekday", "hour_of_day"], how="left")

    model_selected = merged[merged["simulation_permutation"] == merged["recommended_permutation"]]
    model_selected = model_selected.sort_values(["test_start", "hour_of_day"])
    model_curve = model_selected.groupby("test_start", as_index=False)["actual_pnl"].sum()
    model_curve["cumulative_pnl"] = model_curve["actual_pnl"].cumsum()
    model_curve["strategy"] = "Model-selected portfolio"

    baseline = predictions[predictions["simulation_permutation"] == baseline_permutation]
    baseline = baseline.sort_values(["test_start", "hour_of_day"])
    baseline_curve = baseline.groupby("test_start", as_index=False)["actual_pnl"].sum()
    baseline_curve["cumulative_pnl"] = baseline_curve["actual_pnl"].cumsum()
    baseline_curve["strategy"] = f"Baseline (always {baseline_permutation})"

    return model_curve, baseline_curve


def plot_equity_curves(model_curve: pd.DataFrame, baseline_curve: pd.DataFrame, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pd.to_datetime(model_curve["test_start"]), model_curve["cumulative_pnl"],
            label="Model-selected portfolio", linewidth=2, color="#2ca02c")
    ax.plot(pd.to_datetime(baseline_curve["test_start"]), baseline_curve["cumulative_pnl"],
            label=baseline_curve["strategy"].iloc[0], linewidth=2, linestyle="--", color="#7f7f7f")

    ax.set_xlabel("Out-of-sample fold test start date")
    ax.set_ylabel("Cumulative P&L")
    ax.set_title("Model-Selected Strategy Portfolio vs. Baseline (out-of-sample equity curve)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info("Saved equity curve to %s", save_path)


def run(predictions: pd.DataFrame):
    best_matrix = build_best_permutation_matrix(predictions)
    heatmap_path = OUTPUT_DIR / "matrix_heatmap.png"
    plot_heatmap(best_matrix, heatmap_path)

    model_curve, baseline_curve = build_equity_curves(predictions)
    equity_path = OUTPUT_DIR / "equity_curve.png"
    plot_equity_curves(model_curve, baseline_curve, equity_path)

    best_matrix.to_csv(OUTPUT_DIR / "best_permutation_matrix.csv", index=False)
    model_curve.to_csv(OUTPUT_DIR / "model_equity_curve.csv", index=False)
    baseline_curve.to_csv(OUTPUT_DIR / "baseline_equity_curve.csv", index=False)

    return best_matrix, model_curve, baseline_curve


if __name__ == "__main__":
    from db.db_utils import read_sql
    preds = read_sql("SELECT * FROM walk_forward_predictions")
    run(preds)

