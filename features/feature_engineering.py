import logging

import numpy as np
import pandas as pd

from db.db_utils import init_db, read_sql, write_df, execute

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("feature_engineering")

DEDUP_WINDOW_MS = 500
MIN_TRADES_PER_ACCOUNT = 20  


def load_raw() -> pd.DataFrame:
    df = read_sql("SELECT * FROM trading_logs_raw")
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df


def load_macro() -> pd.DataFrame:
    df = read_sql("SELECT * FROM macro_events")
    df["event_ts_utc"] = pd.to_datetime(df["event_ts_utc"], utc=True)
    return df


def dedup_transactions(df: pd.DataFrame, window_ms: int = DEDUP_WINDOW_MS) -> pd.DataFrame:
    
    df = df.sort_values(["account_id", "direction", "quantity", "price", "ts_utc"]).copy()
    df["is_dedup_kept"] = 1
    df["dedup_group_id"] = None

    group_counter = 0

    for (_, _, _, _), group in df.groupby(["account_id", "direction", "quantity", "price"], sort=False):
        group = group.sort_values("ts_utc")
        last_ts = None
        current_group_id = None
        for idx, row in group.iterrows():
            if last_ts is not None and (row["ts_utc"] - last_ts).total_seconds() * 1000 <= window_ms:
                # duplicate of the previous fill -> drop, tag with the group id
                df.loc[idx, "is_dedup_kept"] = 0
                df.loc[idx, "dedup_group_id"] = current_group_id
            else:
                group_counter += 1
                current_group_id = f"grp-{group_counter}"
                df.loc[idx, "dedup_group_id"] = current_group_id
            last_ts = row["ts_utc"]

    n_dropped = (df["is_dedup_kept"] == 0).sum()
    log.info("Deduplication: dropped %d of %d rows as duplicate fills (<%dms apart).",
              n_dropped, len(df), window_ms)
    return df


def account_level_filter(df: pd.DataFrame, min_trades: int = MIN_TRADES_PER_ACCOUNT) -> pd.DataFrame:
    kept = df[df["is_dedup_kept"] == 1].copy()
    counts = kept.groupby("account_id").size()
    valid_accounts = counts[counts >= min_trades].index
    n_dropped_accounts = counts.shape[0] - len(valid_accounts)
    if n_dropped_accounts:
        log.info("Account-level filter: dropping %d accounts with < %d trades.",
                  n_dropped_accounts, min_trades)
    df.loc[~df["account_id"].isin(valid_accounts), "is_dedup_kept"] = 0
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["weekday"] = df["ts_utc"].dt.weekday  # 0=Mon .. 6=Sun
    df["hour_of_day"] = df["ts_utc"].dt.hour
    return df


def add_macro_proximity_features(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    
    macro_sorted = macro.sort_values("event_ts_utc")[
        ["event_ts_utc", "event_name", "surprise_score"]
    ].dropna(subset=["event_ts_utc"])

    trades_sorted = df.sort_values("ts_utc")

    merged = pd.merge_asof(
        trades_sorted,
        macro_sorted,
        left_on="ts_utc",
        right_on="event_ts_utc",
        direction="backward",  
    )

    merged["minutes_since_last_macro_event"] = (
        (merged["ts_utc"] - merged["event_ts_utc"]).dt.total_seconds() / 60.0
    )
    merged = merged.rename(columns={
        "event_name": "nearest_macro_event",
        "surprise_score": "nearest_macro_surprise",
    })
    merged = merged.drop(columns=["event_ts_utc"])
    return merged


def run() -> pd.DataFrame:
    init_db()
    raw = load_raw()
    macro = load_macro()

    if raw.empty:
        raise RuntimeError("No raw trading logs found -- run data ingestion first.")
    if macro.empty:
        log.warning("No macro events found -- macro proximity features will be null.")

    df = dedup_transactions(raw)
    df = account_level_filter(df)
    df = add_time_features(df)

    if not macro.empty:
        df = add_macro_proximity_features(df, macro)
    else:
        df["nearest_macro_event"] = None
        df["nearest_macro_surprise"] = np.nan
        df["minutes_since_last_macro_event"] = np.nan

    clean = df[df["is_dedup_kept"] == 1].copy()
    log.info("Feature engineering complete: %d clean trades (of %d raw).", len(clean), len(raw))

    out_cols = [
        "trade_id", "account_id", "ts_utc", "direction", "quantity", "price", "pnl",
        "simulation_permutation", "is_dedup_kept", "dedup_group_id", "weekday", "hour_of_day",
        "minutes_since_last_macro_event", "nearest_macro_event", "nearest_macro_surprise",
    ]
    store = df[out_cols].copy()
    store["ts_utc"] = store["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")

    execute("DELETE FROM trading_logs_clean")
    write_df(store, "trading_logs_clean", if_exists="append")

    return clean


if __name__ == "__main__":
    result = run()
    print(result.head(10).to_string())