import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from config import (
    APIFY_API_TOKEN,
    APIFY_MACRO_CALENDAR_ACTOR_ID,
    ALLOW_SYNTHETIC_FALLBACK,
    LOOKBACK_DAYS,
    RANDOM_SEED,
)
from db.db_utils import init_db, write_df, execute

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("apify_macro_scraper")

EST_OFFSET_HOURS = -5  

MACRO_EVENT_TEMPLATES = [
    ("CPI m/m", "US", 0.03),
    ("Core CPI m/m", "US", 0.02),
    ("FOMC Statement", "US", None),
    ("Fed Interest Rate Decision", "US", 0.0),
    ("Non-Farm Employment Change", "US", 180.0),
    ("Unemployment Rate", "US", 4.0),
    ("Average Hourly Earnings m/m", "US", 0.3),
    ("Retail Sales m/m", "US", 0.4),
    ("ISM Manufacturing PMI", "US", 50.0),
]


def _utc_to_est(ts_utc: datetime) -> datetime:
    return ts_utc + timedelta(hours=EST_OFFSET_HOURS)


def fetch_via_apify(lookback_days: int) -> pd.DataFrame | None:

    if not APIFY_API_TOKEN:
        log.warning("No APIFY_API_TOKEN configured -- skipping live Apify call.")
        return None

    try:
        from apify_client import ApifyClient
    except ImportError:
        log.warning("apify-client not installed -- skipping live Apify call.")
        return None

    try:
        client = ApifyClient(APIFY_API_TOKEN)
        date_from = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        date_to = datetime.utcnow().strftime("%Y-%m-%d")

        run_input = {
            "dateFrom": date_from,
            "dateTo": date_to,
            "countries": ["United States"],
            "importance": ["high", "medium"],
        }
        log.info("Calling Apify actor '%s' with input %s", APIFY_MACRO_CALENDAR_ACTOR_ID, run_input)
        run = client.actor(APIFY_MACRO_CALENDAR_ACTOR_ID).call(run_input=run_input)
        items = list(client.dataset(run["defaultDatasetId"]).iterate_items())

        if not items:
            log.warning("Apify actor returned 0 items.")
            return None

        df = pd.json_normalize(items)
        log.info("Fetched %d raw macro events from Apify.", len(df))
        return df

    except Exception as exc:  # noqa: BLE001 - we want any failure to trigger fallback
        log.error("Apify call failed: %s", exc)
        return None


def _normalize_apify_df(raw: pd.DataFrame) -> pd.DataFrame:
    
    col_map_candidates = {
        "event_name": ["event", "title", "eventName", "name"],
        "event_ts": ["date", "datetime", "timestamp", "time"],
        "country": ["country", "region"],
        "forecast_value": ["forecast", "forecastValue"],
        "actual_value": ["actual", "actualValue"],
        "previous_value": ["previous", "previousValue"],
    }

    def pick(colnames):
        for c in colnames:
            if c in raw.columns:
                return raw[c]
        return pd.Series([None] * len(raw))

    out = pd.DataFrame({
        "event_name": pick(col_map_candidates["event_name"]),
        "event_ts_utc": pd.to_datetime(pick(col_map_candidates["event_ts"]), utc=True, errors="coerce"),
        "country": pick(col_map_candidates["country"]),
        "forecast_value": pd.to_numeric(pick(col_map_candidates["forecast_value"]), errors="coerce"),
        "actual_value": pd.to_numeric(pick(col_map_candidates["actual_value"]), errors="coerce"),
        "previous_value": pd.to_numeric(pick(col_map_candidates["previous_value"]), errors="coerce"),
    })
    return out.dropna(subset=["event_name", "event_ts_utc"])


def generate_synthetic_macro_calendar(lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
   
    rng = np.random.default_rng(RANDOM_SEED)
    start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    rows = []

    for name, country, base_val in MACRO_EVENT_TEMPLATES:
        if "FOMC" in name or "Interest Rate" in name:
            interval_days = 45
            release_hour_utc = 19  # 2pm ET
        else:
            interval_days = 30
            release_hour_utc = 13  # 8:30am ET, rounded to hour for simplicity

        n_releases = max(1, lookback_days // interval_days)
        for i in range(n_releases):
            event_dt = start + timedelta(days=i * interval_days + int(rng.integers(0, 5)))
            event_dt = event_dt.replace(hour=release_hour_utc, minute=30, second=0, microsecond=0)
            if event_dt > datetime.now(timezone.utc):
                continue

            if base_val is None:
                forecast = np.nan
                actual = np.nan
            else:
                noise_scale = max(abs(base_val) * 0.15, 0.05)
                forecast = round(float(base_val + rng.normal(0, noise_scale)), 3)
                actual = round(float(forecast + rng.normal(0, noise_scale)), 3)

            rows.append({
                "event_name": name,
                "event_ts_utc": event_dt,
                "country": country,
                "forecast_value": forecast,
                "actual_value": actual,
                "previous_value": round(float(base_val), 3) if base_val is not None else np.nan,
            })

    df = pd.DataFrame(rows).sort_values("event_ts_utc").reset_index(drop=True)
    log.info("Generated %d synthetic macro events (fallback mode).", len(df))
    return df


def enrich_and_store(df: pd.DataFrame, source: str):
    df = df.copy()
    df["event_ts_utc"] = pd.to_datetime(df["event_ts_utc"], utc=True)
    df["event_ts_est"] = df["event_ts_utc"].apply(lambda t: _utc_to_est(t.to_pydatetime()))
    df["event_ts_utc"] = df["event_ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    df["event_ts_est"] = df["event_ts_est"].apply(lambda t: t.strftime("%Y-%m-%dT%H:%M:%S"))

    df["surprise_score"] = np.where(
        df["forecast_value"].abs() > 1e-9,
        (df["actual_value"] - df["forecast_value"]) / df["forecast_value"].abs(),
        np.nan,
    )
    df["source"] = source
    df["scraped_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")

    cols = [
        "event_name", "event_ts_utc", "event_ts_est", "country",
        "forecast_value", "actual_value", "previous_value", "surprise_score",
        "source", "scraped_at_utc",
    ]
    write_df(df[cols], "macro_events", if_exists="append")
    log.info("Stored %d macro events into DB (source=%s).", len(df), source)
    return df


def run(lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    init_db()
    execute("DELETE FROM macro_events")  # idempotent re-run for this demo

    raw = fetch_via_apify(lookback_days)
    if raw is not None:
        try:
            df = _normalize_apify_df(raw)
            if df.empty:
                raise ValueError("Normalized Apify frame is empty")
            return enrich_and_store(df, source="apify")
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to normalize live Apify data: %s", exc)

    if not ALLOW_SYNTHETIC_FALLBACK:
        raise RuntimeError("Apify fetch failed and synthetic fallback is disabled.")

    log.warning("FALLING BACK TO SYNTHETIC MACRO CALENDAR (no live Apify data available in this run).")
    synthetic = generate_synthetic_macro_calendar(lookback_days)
    return enrich_and_store(synthetic, source="synthetic_fallback")


if __name__ == "__main__":
    result = run()
    print(result.head(10).to_string())