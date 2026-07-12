"""
End-to-end pipeline orchestrator.

    python main.py

Runs, in order:
  1. Macro event ingestion (Apify, with synthetic fallback)
  2. Synthetic trading log generation  <-- swap for your real log loader
  3. Cleaning + feature engineering
  4. Walk-forward validated model training
  5. Matrix (heatmap) + equity curve generation
  6. Evaluation report (Markdown)

All intermediate artifacts land in db/cwt_pipeline.sqlite and
output/artifacts/. The final report lands in reports/evaluation_report.md
"""
import logging
from datetime import datetime, timezone

import pandas as pd

from config import REPORTS_DIR, OUTPUT_DIR
from scraping import apify_macro_scraper
from data import synthetic_trading_logs
from features import feature_engineering
from model import walk_forward
from output import matrix_generator, eval_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("main")


def write_evaluation_report(metrics_df: pd.DataFrame, model_summary: dict, baseline_summary: dict,
                             n_folds: int, n_predictions: int):
    report_path = REPORTS_DIR / "evaluation_report.md"

    lines = []
    lines.append("# CrowdWisdomTrading Quant Pipeline -- Evaluation Report\n")
    lines.append(f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n")

    lines.append("## Validation design\n")
    lines.append(
        f"- Walk-forward validation: train on {30} days, test on the next {7} days, "
        f"slide forward by {7} days.\n"
        f"- Total folds evaluated: **{n_folds}**\n"
        f"- Total out-of-sample (hour x weekday x permutation) bucket predictions: **{n_predictions}**\n"
        "- All macro-event features are joined via a backward `merge_asof` -- a trade can only ever "
        "see macro events that had *already occurred* at its own timestamp, so there is no look-ahead bias.\n"
        "- Each fold's model is trained only on data strictly before its test window; test windows never "
        "overlap with their own training window.\n"
    )

    lines.append("## Per-fold model fit (MAE / R2 on held-out bucket-level mean P&L)\n")
    lines.append(metrics_df.to_markdown(index=False))
    lines.append("\n")

    lines.append("## Portfolio performance (out-of-sample, weekly folds)\n")
    perf_table = pd.DataFrame([model_summary, baseline_summary])
    lines.append(perf_table.to_markdown(index=False))
    lines.append("\n")

    lines.append(
        "\n_Note: Sortino ratio shows as NaN when a strategy has fewer than 2 losing periods in the "
        "sample, since the downside-deviation denominator is undefined/unstable on that little data -- "
        "not a bug, just a small-sample artifact that will resolve with a longer backtest history._\n"
    )

    lines.append("## Key artifacts\n")
    lines.append("- `output/artifacts/matrix_heatmap.png` -- best simulation permutation per hour x weekday cell\n")
    lines.append("- `output/artifacts/equity_curve.png` -- model-selected portfolio vs. baseline\n")
    lines.append("- `output/artifacts/best_permutation_matrix.csv`\n")
    lines.append("- `db/cwt_pipeline.sqlite` -- full structured DB (macro_events, trading_logs_raw, "
                  "trading_logs_clean, walk_forward_predictions)\n")

    lines.append("\n## Notes on this run\n")
    lines.append(
        "This run used synthetic trading-log data (see `data/synthetic_trading_logs.py`) because no real "
        "trading log export was provided. Swap that module for a real loader against your actual trade "
        "blotter to run this on live data -- the DB schema, cleaning, feature engineering, and "
        "walk-forward model are all data-agnostic and require no changes.\n"
    )

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    log.info("Wrote evaluation report to %s", report_path)
    return report_path


def run_pipeline():
    log.info("=== STEP 1: Macro event ingestion (Apify) ===")
    apify_macro_scraper.run()

    log.info("=== STEP 2: Trading log ingestion ===")
    synthetic_trading_logs.run()

    log.info("=== STEP 3: Cleaning + feature engineering ===")
    feature_engineering.run()

    log.info("=== STEP 4: Walk-forward validated model training ===")
    predictions, metrics_df = walk_forward.run()

    log.info("=== STEP 5: Matrix + equity curve generation ===")
    best_matrix, model_curve, baseline_curve = matrix_generator.run(predictions)

    log.info("=== STEP 6: Evaluation report ===")
    model_summary = eval_metrics.summarize(model_curve, "Model-selected portfolio")
    baseline_summary = eval_metrics.summarize(baseline_curve, "Baseline (single default permutation)")
    report_path = write_evaluation_report(
        metrics_df, model_summary, baseline_summary,
        n_folds=metrics_df["fold_id"].nunique(), n_predictions=len(predictions),
    )

    log.info("=== PIPELINE COMPLETE ===")
    log.info("Model portfolio Sharpe=%.2f Sortino=%.2f MaxDD=%.2f | Baseline Sharpe=%.2f Sortino=%.2f MaxDD=%.2f",
              model_summary["sharpe_ratio"], model_summary["sortino_ratio"], model_summary["max_drawdown"],
              baseline_summary["sharpe_ratio"], baseline_summary["sortino_ratio"], baseline_summary["max_drawdown"])

    return {
        "predictions": predictions,
        "metrics_df": metrics_df,
        "best_matrix": best_matrix,
        "model_curve": model_curve,
        "baseline_curve": baseline_curve,
        "report_path": report_path,
    }


if __name__ == "__main__":
    run_pipeline()