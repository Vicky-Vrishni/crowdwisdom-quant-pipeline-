import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from config import LOOKBACK_DAYS, N_ACCOUNTS, SIMULATION_PERMUTATIONS, RANDOM_SEED
from db.db_utils import init_db, write_df, execute, read_sql

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("synthetic_trading_logs")

PERMUTATION_PROFILES = {
    "A_scalp_5m":      {"base_pnl": 1.0,  "pnl_std": 6, "best_hour_start": 13, "best_hour_end": 16, "hour_bonus": 6.0, "macro_sensitivity": -3.0},
    "B_trend_15m":     {"base_pnl": 6.0,  "pnl_std": 7, "best_hour_start": 16, "best_hour_end": 20, "hour_bonus": 7.0, "macro_sensitivity": 1.0},
    "C_meanrev_1h":    {"base_pnl": -2.0, "pnl_std": 6, "best_hour_start": 7,  "best_hour_end": 11, "hour_bonus": 5.0, "macro_sensitivity": -4.0},
    "D_breakout_30m":  {"base_pnl": 2.0,  "pnl_std": 7, "best_hour_start": 19, "best_hour_end": 22, "hour_bonus": 6.5, "macro_sensitivity": 3.0},
    "E_news_fade_5m":  {"base_pnl": -3.0, "pnl_std": 6, "best_hour_start": 13, "best_hour_end": 14, "hour_bonus": 4.0, "macro_sensitivity": 9.0},
}

MACRO_INFLUENCE_WINDOW_MIN = 90


def _load_macro_events():
    macro = read_sql("SELECT event_ts_utc, surprise_score FROM macro_events")
    if macro.empty:
        return macro
    macro["event_ts_utc"] = pd.to_datetime(macro["event_ts_utc"], utc=True)
    macro = macro.dropna(subset=["surprise_score"]).sort_values("event_ts_utc").reset_index(drop=True)
    return macro


def _nearest_past_surprise(ts, macro_index: pd.DatetimeIndex, macro_surprises: np.ndarray):
    if len(macro_index) == 0:
        return 0.0
    idx = macro_index.searchsorted(ts, side="right") - 1
    if idx < 0:
        return 0.0
    event_ts = macro_index[idx]
    minutes_elapsed = (ts - event_ts).total_seconds() / 60.0
    if minutes_elapsed <= MACRO_INFLUENCE_WINDOW_MIN:
        return float(macro_surprises[idx])
    return 0.0


def generate(lookback_days: int = LOOKBACK_DAYS, n_accounts: int = N_ACCOUNTS,
             trades_per_day_per_account: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    rows = []

    macro = _load_macro_events()
    macro_ts = pd.DatetimeIndex(macro["event_ts_utc"]) if not macro.empty else pd.DatetimeIndex([])
    macro_surprise = macro["surprise_score"].to_numpy() if not macro.empty else np.array([])
    if macro.empty:
        log.warning("No macro events found in DB -- run the macro scraper BEFORE the trading log "
                    "generator, or macro-sensitivity signal will be flat. Falling back to 0 surprise.")

    for day_offset in range(lookback_days):
        day = start + timedelta(days=day_offset)
        if day.weekday() >= 5:  # skip weekends, markets closed
            continue

        for account_idx in range(n_accounts):
            account_id = f"ACC-{account_idx+1:03d}"
            n_trades = rng.poisson(trades_per_day_per_account)

            for _ in range(n_trades):
                hour = int(rng.choice(range(0, 24), p=_hour_weights()))
                minute = int(rng.integers(0, 60))
                second = int(rng.integers(0, 60))
                ts = day.replace(hour=hour, minute=minute, second=second, microsecond=0)

                perm = rng.choice(SIMULATION_PERMUTATIONS)
                profile = PERMUTATION_PROFILES[perm]
                direction = rng.choice(["long", "short"])
                quantity = float(rng.integers(1, 10))
                price = round(float(rng.uniform(50, 500)), 2)

                expected_pnl = profile["base_pnl"]

                if profile["best_hour_start"] <= hour < profile["best_hour_end"]:
                    expected_pnl += profile["hour_bonus"]

                surprise = _nearest_past_surprise(
                    pd.Timestamp(ts), macro_ts, macro_surprise
                )
                expected_pnl += profile["macro_sensitivity"] * surprise

                pnl = float(rng.normal(expected_pnl, profile["pnl_std"]))

                rows.append({
                    "account_id": account_id,
                    "ts_utc": ts,
                    "direction": direction,
                    "quantity": quantity,
                    "price": price,
                    "pnl": round(pnl, 2),
                    "simulation_permutation": perm,
                })

                if rng.random() < 0.3:
                    dup_ts = ts + timedelta(milliseconds=int(rng.integers(50, 480)))
                    rows.append({
                        "account_id": account_id,
                        "ts_utc": dup_ts,
                        "direction": direction,
                        "quantity": quantity,
                        "price": price,
                        "pnl": round(pnl, 2),
                        "simulation_permutation": perm,
                    })

    df = pd.DataFrame(rows).sort_values("ts_utc").reset_index(drop=True)
    log.info("Generated %d synthetic raw trade rows (including injected near-duplicates).", len(df))
    return df


def _hour_weights():
    """Weight trading activity toward US market hours (13:30-20:00 UTC)
    with a smaller Asia/London session bump, so time-of-day features are
    meaningful."""
    weights = np.ones(24)
    weights[13:20] = 4.0   # US session
    weights[7:11] = 1.5    # London session
    weights[0:3] = 1.2     # Asia session
    return weights / weights.sum()


def store(df: pd.DataFrame):
    df = df.copy()
    df["ts_utc"] = df["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    df["raw_ingested_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
    write_df(df, "trading_logs_raw", if_exists="append")
    log.info("Stored %d raw trade rows into DB.", len(df))


def run():
    init_db()
    execute("DELETE FROM trading_logs_raw")
    df = generate()
    store(df)
    return df


if __name__ == "__main__":
    result = run()
    print(result.head(10).to_string())