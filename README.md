# CrowdWisdomTrading -- Quant Data Science Pipeline

A Python ML pipeline that ingests historical trading logs, scrapes a
macroeconomic event calendar via Apify, joins both datasets in SQLite,
and trains a walk-forward-validated model that recommends the best
simulation permutation for each (weekday, hour-of-day) trading slot.

## What this does

1. **Macro Event Ingestion** (`scraping/apify_macro_scraper.py`)
   Scrapes CPI / FOMC / employment releases for the last 180 days via
   an Apify actor, normalizes timestamps to UTC + EST, computes a
   `surprise_score = (actual - forecast) / |forecast|`, and stores it
   in SQLite.

2. **Trading Log Ingestion + Cleaning** (`features/feature_engineering.py`)
   Deduplicates transactions reported <500ms apart on the same account
   (same direction/qty/price), applies an account-level minimum-trade
   filter, and joins each trade to time-based features (`weekday`,
   `hour_of_day`) and macro-proximity features (`minutes_since_last_macro_event`,
   `nearest_macro_surprise`) via a **backward `merge_asof`** -- so a
   trade only ever "sees" macro events that already happened. This is
   what prevents look-ahead leakage at the feature level.

3. **Walk-Forward Validated Model** (`model/walk_forward.py`)
   Aggregates clean trades into `(date, permutation, hour, weekday)`
   buckets and trains an XGBoost regressor per fold using a sliding
   window: **train 30 days -> test next 7 days -> slide 7 days ->
   repeat**. The model in fold *N* never sees data from fold *N*'s own
   test window or anything after it.

4. **Matrix + Equity Curve Outputs** (`output/matrix_generator.py`)
   - A heatmap of the best-predicted simulation permutation for every
     `(weekday, hour_of_day)` cell.
   - An equity curve comparing a "model-selected portfolio" (trades
     whatever permutation the model recommended for each out-of-sample
     bucket, booking its *actual* realized P&L) against a naive
     baseline that always runs a single fixed permutation.

5. **Evaluation Report** (`reports/evaluation_report.md`)
   Sharpe ratio, Sortino ratio, max drawdown, and per-fold MAE/R2,
   auto-generated after every run.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Required for live macro scraping -- get a token at https://console.apify.com/account/integrations
export APIFY_API_TOKEN="apify_api_xxx"        # Windows: $env:APIFY_API_TOKEN="apify_api_xxx"

# Optional: point at a specific Apify actor for the calendar scrape
export APIFY_MACRO_CALENDAR_ACTOR_ID="your-username/your-actor"
```

## Run

```bash
python main.py
```

Outputs:
- `db/cwt_pipeline.sqlite` -- all structured data (raw + clean trades, macro events, predictions)
- `output/artifacts/matrix_heatmap.png`
- `output/artifacts/equity_curve.png`
- `output/artifacts/best_permutation_matrix.csv`
- `reports/evaluation_report.md`

## Using real trading log data

This repo ships with `data/synthetic_trading_logs.py`, a generator that
produces a realistic but fake trade blotter (including injected
near-duplicate fills and genuine hour-of-day / macro-surprise edges per
permutation) purely so the pipeline can be demonstrated end-to-end
without access to a real account's logs.

**To run this on real data:** replace the call to
`synthetic_trading_logs.run()` in `main.py` Step 2 with your own loader
that reads your actual trade export (CSV, broker API, internal DB, etc.)
and writes it into the `trading_logs_raw` table using the same schema
(see `db/schema.sql`). Nothing downstream (cleaning, feature
engineering, walk-forward model, matrix/equity generation) needs to
change -- it's all written against the schema, not the data source.

## Design notes / why things are built this way

- **No look-ahead bias, twice over.** Once at the feature-join level
  (backward `merge_asof` on macro events), and once at the model-training
  level (walk-forward folds where train always strictly precedes test).
- **Deduplication window is configurable** (`DEDUP_WINDOW_MS` in
  `features/feature_engineering.py`), defaulting to the 500ms window
  specified in the assignment brief.
- **Apify failures degrade gracefully.** If no token is configured, the
  actor call fails, or the network is unreachable, the pipeline logs a
  loud warning and falls back to a synthetic macro calendar so the rest
  of the pipeline can still run end-to-end. Set
  `ALLOW_SYNTHETIC_FALLBACK = False` in `config.py` to disable this and
  force a hard failure instead.
- **Surprise score** (`(actual - forecast) / |forecast|`) is the
  headline "macro surprise" feature called out in the evaluation
  criteria -- it's what lets the model learn that certain permutations
  (e.g. a news-fade strategy) behave differently right after a high-surprise
  CPI print vs. a quiet data day.

## Repo structure

```
config.py                      # all paths / DB / Apify / walk-forward settings
db/
  schema.sql                   # SQLite schema
  db_utils.py                  # thin sqlite3 helper layer
scraping/
  apify_macro_scraper.py       # Apify macro calendar scraper (+ synthetic fallback)
data/
  synthetic_trading_logs.py    # demo trade blotter generator (swap for real loader)
features/
  feature_engineering.py       # dedup, account filter, time + macro features
model/
  walk_forward.py              # XGBoost + walk-forward validation
output/
  matrix_generator.py          # heatmap + equity curve
  eval_metrics.py              # Sharpe / Sortino / Max Drawdown
reports/
  evaluation_report.md         # auto-generated after each run
main.py                        # orchestrates the full pipeline
```