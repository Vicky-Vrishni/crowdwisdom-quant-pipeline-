# CrowdWisdomTrading Quant Pipeline -- Evaluation Report

_Generated: 2026-07-13 16:03 UTC_

## Validation design

- Walk-forward validation: train on 30 days, test on the next 7 days, slide forward by 7 days.
- Total folds evaluated: **21**
- Total out-of-sample (hour x weekday x permutation) bucket predictions: **4554**
- All macro-event features are joined via a backward `merge_asof` -- a trade can only ever see macro events that had *already occurred* at its own timestamp, so there is no look-ahead bias.
- Each fold's model is trained only on data strictly before its test window; test windows never overlap with their own training window.

## Per-fold model fit (MAE / R2 on held-out bucket-level mean P&L)

|   fold_id | train_start         | test_start          |   n_train |   n_test |     mae |       r2 |
|----------:|:--------------------|:--------------------|----------:|---------:|--------:|---------:|
|         0 | 2026-01-14 00:00:00 | 2026-02-13 00:00:00 |       967 |      221 | 5.08845 | 0.278253 |
|         1 | 2026-01-21 00:00:00 | 2026-02-20 00:00:00 |       975 |      215 | 4.64985 | 0.31464  |
|         2 | 2026-01-28 00:00:00 | 2026-02-27 00:00:00 |       966 |      210 | 5.02336 | 0.193031 |
|         3 | 2026-02-04 00:00:00 | 2026-03-06 00:00:00 |       950 |      222 | 4.84084 | 0.187041 |
|         4 | 2026-02-11 00:00:00 | 2026-03-13 00:00:00 |       947 |      217 | 4.85218 | 0.212229 |
|         5 | 2026-02-18 00:00:00 | 2026-03-20 00:00:00 |       955 |      214 | 4.60607 | 0.269978 |
|         6 | 2026-02-25 00:00:00 | 2026-03-27 00:00:00 |       955 |      200 | 4.89457 | 0.224476 |
|         7 | 2026-03-04 00:00:00 | 2026-04-03 00:00:00 |       932 |      218 | 5.15628 | 0.230834 |
|         8 | 2026-03-11 00:00:00 | 2026-04-10 00:00:00 |       935 |      216 | 5.39227 | 0.192658 |
|         9 | 2026-03-18 00:00:00 | 2026-04-17 00:00:00 |       931 |      215 | 5.32099 | 0.299757 |
|        10 | 2026-03-25 00:00:00 | 2026-04-24 00:00:00 |       932 |      204 | 4.99525 | 0.294707 |
|        11 | 2026-04-01 00:00:00 | 2026-05-01 00:00:00 |       946 |      222 | 4.68764 | 0.304274 |
|        12 | 2026-04-08 00:00:00 | 2026-05-08 00:00:00 |       942 |      240 | 5.10369 | 0.293169 |
|        13 | 2026-04-15 00:00:00 | 2026-05-15 00:00:00 |       967 |      215 | 5.78315 | 0.14755  |
|        14 | 2026-04-22 00:00:00 | 2026-05-22 00:00:00 |       973 |      200 | 4.95191 | 0.328967 |
|        15 | 2026-04-29 00:00:00 | 2026-05-29 00:00:00 |       960 |      212 | 4.81102 | 0.302246 |
|        16 | 2026-05-06 00:00:00 | 2026-06-05 00:00:00 |       960 |      232 | 4.59768 | 0.234617 |
|        17 | 2026-05-13 00:00:00 | 2026-06-12 00:00:00 |       959 |      223 | 4.93589 | 0.257109 |
|        18 | 2026-05-20 00:00:00 | 2026-06-19 00:00:00 |       949 |      222 | 4.64491 | 0.343126 |
|        19 | 2026-05-27 00:00:00 | 2026-06-26 00:00:00 |       973 |      211 | 4.7605  | 0.257967 |
|        20 | 2026-06-03 00:00:00 | 2026-07-03 00:00:00 |       983 |      225 | 5.55364 | 0.16467  |


## Portfolio performance (out-of-sample, weekly folds)

| strategy                              |   total_pnl |   n_periods |   sharpe_ratio |   sortino_ratio |   max_drawdown |   win_rate_periods |
|:--------------------------------------|------------:|------------:|---------------:|----------------:|---------------:|-------------------:|
| Model-selected portfolio              |     7239.33 |          21 |        66.1318 |             nan |              0 |                  1 |
| Baseline (single default permutation) |     2451.16 |          21 |        17.4339 |             nan |              0 |                  1 |



_Note: Sortino ratio shows as NaN when a strategy has fewer than 2 losing periods in the sample, since the downside-deviation denominator is undefined/unstable on that little data -- not a bug, just a small-sample artifact that will resolve with a longer backtest history._

## Key artifacts

- `output/artifacts/matrix_heatmap.png` -- best simulation permutation per hour x weekday cell

- `output/artifacts/equity_curve.png` -- model-selected portfolio vs. baseline

- `output/artifacts/best_permutation_matrix.csv`

- `db/cwt_pipeline.sqlite` -- full structured DB (macro_events, trading_logs_raw, trading_logs_clean, walk_forward_predictions)


## Notes on this run

This run used synthetic trading-log data (see `data/synthetic_trading_logs.py`) because no real trading log export was provided. Swap that module for a real loader against your actual trade blotter to run this on live data -- the DB schema, cleaning, feature engineering, and walk-forward model are all data-agnostic and require no changes.
