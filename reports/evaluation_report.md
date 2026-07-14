# CrowdWisdomTrading Quant Pipeline -- Evaluation Report

_Generated: 2026-07-14 07:25 UTC_

## Validation design

- Walk-forward validation: train on 30 days, test on the next 7 days, slide forward by 7 days.
- Total folds evaluated: **21**
- Total out-of-sample (hour x weekday x permutation) bucket predictions: **4554**
- All macro-event features are joined via a backward `merge_asof` -- a trade can only ever see macro events that had *already occurred* at its own timestamp, so there is no look-ahead bias.
- Each fold's model is trained only on data strictly before its test window; test windows never overlap with their own training window.

## Per-fold model fit (MAE / R2 on held-out bucket-level mean P&L)

|   fold_id | train_start         | test_start          |   n_train |   n_test |     mae |        r2 |
|----------:|:--------------------|:--------------------|----------:|---------:|--------:|----------:|
|         0 | 2026-01-15 00:00:00 | 2026-02-14 00:00:00 |       967 |      221 | 5.23316 | 0.253108  |
|         1 | 2026-01-22 00:00:00 | 2026-02-21 00:00:00 |       975 |      215 | 4.57573 | 0.32836   |
|         2 | 2026-01-29 00:00:00 | 2026-02-28 00:00:00 |       966 |      210 | 4.98559 | 0.212191  |
|         3 | 2026-02-05 00:00:00 | 2026-03-07 00:00:00 |       950 |      222 | 4.8258  | 0.187092  |
|         4 | 2026-02-12 00:00:00 | 2026-03-14 00:00:00 |       947 |      217 | 5.43685 | 0.0840158 |
|         5 | 2026-02-19 00:00:00 | 2026-03-21 00:00:00 |       955 |      214 | 4.68173 | 0.254701  |
|         6 | 2026-02-26 00:00:00 | 2026-03-28 00:00:00 |       955 |      200 | 4.69723 | 0.263087  |
|         7 | 2026-03-05 00:00:00 | 2026-04-04 00:00:00 |       932 |      218 | 5.08201 | 0.249712  |
|         8 | 2026-03-12 00:00:00 | 2026-04-11 00:00:00 |       935 |      216 | 5.59079 | 0.134874  |
|         9 | 2026-03-19 00:00:00 | 2026-04-18 00:00:00 |       931 |      215 | 5.30352 | 0.297085  |
|        10 | 2026-03-26 00:00:00 | 2026-04-25 00:00:00 |       932 |      204 | 4.97094 | 0.292901  |
|        11 | 2026-04-02 00:00:00 | 2026-05-02 00:00:00 |       946 |      222 | 4.67307 | 0.301934  |
|        12 | 2026-04-09 00:00:00 | 2026-05-09 00:00:00 |       942 |      240 | 4.99417 | 0.307234  |
|        13 | 2026-04-16 00:00:00 | 2026-05-16 00:00:00 |       967 |      215 | 5.89368 | 0.152383  |
|        14 | 2026-04-23 00:00:00 | 2026-05-23 00:00:00 |       973 |      200 | 4.82361 | 0.350235  |
|        15 | 2026-04-30 00:00:00 | 2026-05-30 00:00:00 |       960 |      212 | 4.82631 | 0.326214  |
|        16 | 2026-05-07 00:00:00 | 2026-06-06 00:00:00 |       960 |      232 | 4.52788 | 0.238161  |
|        17 | 2026-05-14 00:00:00 | 2026-06-13 00:00:00 |       959 |      223 | 4.92844 | 0.228127  |
|        18 | 2026-05-21 00:00:00 | 2026-06-20 00:00:00 |       949 |      222 | 4.70694 | 0.337607  |
|        19 | 2026-05-28 00:00:00 | 2026-06-27 00:00:00 |       973 |      211 | 4.87902 | 0.213516  |
|        20 | 2026-06-04 00:00:00 | 2026-07-04 00:00:00 |       983 |      225 | 5.3884  | 0.205322  |


## Portfolio performance (out-of-sample, weekly folds)

| strategy                              |   total_pnl |   n_periods |   sharpe_ratio |   sortino_ratio |   max_drawdown |   win_rate_periods |
|:--------------------------------------|------------:|------------:|---------------:|----------------:|---------------:|-------------------:|
| Model-selected portfolio              |     7218.74 |          21 |        61.379  |             nan |              0 |                  1 |
| Baseline (single default permutation) |     2434.51 |          21 |        17.1504 |             nan |              0 |                  1 |



_Note: Sortino ratio shows as NaN when a strategy has fewer than 2 losing periods in the sample, since the downside-deviation denominator is undefined/unstable on that little data -- not a bug, just a small-sample artifact that will resolve with a longer backtest history._

## Key artifacts

- `output/artifacts/matrix_heatmap.png` -- best simulation permutation per hour x weekday cell

- `output/artifacts/equity_curve.png` -- model-selected portfolio vs. baseline

- `output/artifacts/best_permutation_matrix.csv`

- `db/cwt_pipeline.sqlite` -- full structured DB (macro_events, trading_logs_raw, trading_logs_clean, walk_forward_predictions)


## Notes on this run

This run used synthetic trading-log data (see `data/synthetic_trading_logs.py`) because no real trading log export was provided. Swap that module for a real loader against your actual trade blotter to run this on live data -- the DB schema, cleaning, feature engineering, and walk-forward model are all data-agnostic and require no changes.
