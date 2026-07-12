import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from config import TRAIN_WINDOW_DAYS, TEST_WINDOW_DAYS, STEP_DAYS, RANDOM_SEED
from db.db_utils import init_db, read_sql, write_df, execute

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("walk_forward")

FEATURE_COLS = [
    "hour_of_day", "weekday", "perm_encoded",
    "mean_minutes_since_macro", "mean_macro_surprise", "n_trades",
]


def load_clean_trades() -> pd.DataFrame:
    df = read_sql("SELECT * FROM trading_logs_clean")
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["date"] = df["ts_utc"].dt.date
    return df


def build_daily_buckets(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(
        ["date", "simulation_permutation", "hour_of_day", "weekday"], as_index=False
    ).agg(
        mean_pnl=("pnl", "mean"),
        win_rate=("pnl", lambda x: float((x > 0).mean())),
        n_trades=("pnl", "size"),
        mean_minutes_since_macro=("minutes_since_last_macro_event", "mean"),
        mean_macro_surprise=("nearest_macro_surprise", "mean"),
    )
    grouped["mean_macro_surprise"] = grouped["mean_macro_surprise"].fillna(0.0)
    grouped["mean_minutes_since_macro"] = grouped["mean_minutes_since_macro"].fillna(
        grouped["mean_minutes_since_macro"].median()
    )

    perm_categories = sorted(grouped["simulation_permutation"].unique())
    perm_map = {p: i for i, p in enumerate(perm_categories)}
    grouped["perm_encoded"] = grouped["simulation_permutation"].map(perm_map)

    grouped["date"] = pd.to_datetime(grouped["date"])
    return grouped.sort_values("date").reset_index(drop=True), perm_map


def walk_forward_splits(dates: pd.Series, train_days: int, test_days: int, step_days: int):
    """Yields (train_start, train_end, test_start, test_end) as pd.Timestamps."""
    min_date, max_date = dates.min(), dates.max()
    train_start = min_date
    while True:
        train_end = train_start + pd.Timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=test_days)
        if test_end > max_date + pd.Timedelta(days=1):
            break
        yield train_start, train_end, test_start, test_end
        train_start = train_start + pd.Timedelta(days=step_days)


def run() -> pd.DataFrame:
    init_db()
    trades = load_clean_trades()
    if trades.empty:
        raise RuntimeError("No clean trades found -- run feature engineering first.")

    buckets, perm_map = build_daily_buckets(trades)
    inv_perm_map = {v: k for k, v in perm_map.items()}

    all_predictions = []
    fold_metrics = []

    for fold_id, (train_start, train_end, test_start, test_end) in enumerate(
        walk_forward_splits(buckets["date"], TRAIN_WINDOW_DAYS, TEST_WINDOW_DAYS, STEP_DAYS)
    ):
        train_mask = (buckets["date"] >= train_start) & (buckets["date"] < train_end)
        test_mask = (buckets["date"] >= test_start) & (buckets["date"] < test_end)

        train_df = buckets.loc[train_mask]
        test_df = buckets.loc[test_mask]

        if len(train_df) < 30 or test_df.empty:
            continue  # not enough data in this window yet

        model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            objective="reg:squarederror",
        )
        model.fit(train_df[FEATURE_COLS], train_df["mean_pnl"])

        preds = model.predict(test_df[FEATURE_COLS])
        fold_result = test_df.copy()
        fold_result["predicted_pnl"] = preds
        fold_result["fold_id"] = fold_id
        fold_result["train_start"] = train_start
        fold_result["train_end"] = train_end
        fold_result["test_start"] = test_start
        fold_result["test_end"] = test_end
        all_predictions.append(fold_result)

        mae = mean_absolute_error(test_df["mean_pnl"], preds)
        r2 = r2_score(test_df["mean_pnl"], preds) if test_df["mean_pnl"].nunique() > 1 else np.nan
        fold_metrics.append({"fold_id": fold_id, "train_start": train_start, "test_start": test_start,
                              "n_train": len(train_df), "n_test": len(test_df), "mae": mae, "r2": r2})
        log.info("Fold %d | train[%s -> %s] test[%s -> %s] | n_train=%d n_test=%d MAE=%.3f R2=%.3f",
                  fold_id, train_start.date(), train_end.date(), test_start.date(), test_end.date(),
                  len(train_df), len(test_df), mae, r2 if not np.isnan(r2) else -999)

    if not all_predictions:
        raise RuntimeError(
            "No walk-forward folds produced -- not enough historical days "
            f"for TRAIN_WINDOW_DAYS={TRAIN_WINDOW_DAYS} + TEST_WINDOW_DAYS={TEST_WINDOW_DAYS}."
        )

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions["simulation_permutation"] = predictions["perm_encoded"].map(inv_perm_map)
    predictions = predictions.rename(columns={"mean_pnl": "actual_pnl"})

    store_cols = [
        "fold_id", "train_start", "train_end", "test_start", "test_end",
        "simulation_permutation", "hour_of_day", "weekday",
        "predicted_pnl", "actual_pnl", "n_trades",
    ]
    to_store = predictions[store_cols].copy()
    for c in ["train_start", "train_end", "test_start", "test_end"]:
        to_store[c] = to_store[c].astype(str)

    execute("DELETE FROM walk_forward_predictions")
    write_df(to_store, "walk_forward_predictions", if_exists="append")

    metrics_df = pd.DataFrame(fold_metrics)
    log.info("Walk-forward complete: %d folds, %d out-of-sample bucket predictions.",
              len(metrics_df), len(predictions))

    return predictions, metrics_df


if __name__ == "__main__":
    preds, metrics = run()
    print(metrics.to_string())