# CrowdWisdomTrading Quant Pipeline -- Evaluation Report

_Generated: 2026-07-12 16:15 UTC_

## Validation design

- Walk-forward validation: train on 30 days, test on the next 7 days, slide forward by 7 days.
- Total folds evaluated: **21**
- Total out-of-sample (hour x weekday x permutation) bucket predictions: **4554**
- All macro-event features are joined via a backward `merge_asof` -- a trade can only ever see macro events that had *already occurred* at its own timestamp, so there is no look-ahead bias.
- Each fold's model is trained only on data strictly before its test window; test windows never overlap with their own training window.

## Per-fold model fit (MAE / R2 on held-out bucket-level mean P&L)

|   fold_id | train_start         | test_start          |   n_train |   n_test |     mae |        r2 |
|----------:|:--------------------|:--------------------|----------:|---------:|--------:|----------:|
|         0 | 2026-01-13 00:00:00 | 2026-02-12 00:00:00 |       967 |      221 | 5.24699 |  0.225572 |
|         1 | 2026-01-20 00:00:00 | 2026-02-19 00:00:00 |       975 |      215 | 4.60791 |  0.31345  |
|         2 | 2026-01-27 00:00:00 | 2026-02-26 00:00:00 |       966 |      210 | 5.07881 |  0.170948 |
|         3 | 2026-02-03 00:00:00 | 2026-03-05 00:00:00 |       950 |      222 | 4.77309 |  0.205121 |
|         4 | 2026-02-10 00:00:00 | 2026-03-12 00:00:00 |       947 |      217 | 4.84416 |  0.206791 |
|         5 | 2026-02-17 00:00:00 | 2026-03-19 00:00:00 |       955 |      214 | 4.64401 |  0.270969 |
|         6 | 2026-02-24 00:00:00 | 2026-03-26 00:00:00 |       955 |      200 | 6.65477 | -0.423219 |
|         7 | 2026-03-03 00:00:00 | 2026-04-02 00:00:00 |       932 |      218 | 5.01357 |  0.260303 |
|         8 | 2026-03-10 00:00:00 | 2026-04-09 00:00:00 |       935 |      216 | 5.40215 |  0.195696 |
|         9 | 2026-03-17 00:00:00 | 2026-04-16 00:00:00 |       931 |      215 | 5.4161  |  0.284252 |
|        10 | 2026-03-24 00:00:00 | 2026-04-23 00:00:00 |       932 |      204 | 4.95457 |  0.299145 |
|        11 | 2026-03-31 00:00:00 | 2026-04-30 00:00:00 |       946 |      222 | 4.69928 |  0.288598 |
|        12 | 2026-04-07 00:00:00 | 2026-05-07 00:00:00 |       942 |      240 | 5.13061 |  0.295989 |
|        13 | 2026-04-14 00:00:00 | 2026-05-14 00:00:00 |       967 |      215 | 5.82138 |  0.128297 |
|        14 | 2026-04-21 00:00:00 | 2026-05-21 00:00:00 |       973 |      200 | 5.02674 |  0.313699 |
|        15 | 2026-04-28 00:00:00 | 2026-05-28 00:00:00 |       960 |      212 | 4.74288 |  0.332724 |
|        16 | 2026-05-05 00:00:00 | 2026-06-04 00:00:00 |       960 |      232 | 4.50063 |  0.254229 |
|        17 | 2026-05-12 00:00:00 | 2026-06-11 00:00:00 |       959 |      223 | 4.96026 |  0.250755 |
|        18 | 2026-05-19 00:00:00 | 2026-06-18 00:00:00 |       949 |      222 | 4.69472 |  0.337598 |
|        19 | 2026-05-26 00:00:00 | 2026-06-25 00:00:00 |       973 |      211 | 4.4814  |  0.320108 |
|        20 | 2026-06-02 00:00:00 | 2026-07-02 00:00:00 |       983 |      225 | 5.39384 |  0.208709 |


## Portfolio performance (out-of-sample, weekly folds)

| strategy                              |   total_pnl |   n_periods |   sharpe_ratio |   sortino_ratio |   max_drawdown |   win_rate_periods |
|:--------------------------------------|------------:|------------:|---------------:|----------------:|---------------:|-------------------:|
| Model-selected portfolio              |     7181.37 |          21 |        64.2582 |             nan |              0 |                  1 |
| Baseline (single default permutation) |     2454.86 |          21 |        17.2781 |             nan |              0 |                  1 |



_Note: Sortino ratio shows as NaN when a strategy has fewer than 2 losing periods in the sample, since the downside-deviation denominator is undefined/unstable on that little data -- not a bug, just a small-sample artifact that will resolve with a longer backtest history._

## Key artifacts

- `output/artifacts/matrix_heatmap.png` -- best simulation permutation per hour x weekday cell

- `output/artifacts/equity_curve.png` -- model-selected portfolio vs. baseline

- `output/artifacts/best_permutation_matrix.csv`

- `db/cwt_pipeline.sqlite` -- full structured DB (macro_events, trading_logs_raw, trading_logs_clean, walk_forward_predictions)


## Notes on this run

This run used synthetic trading-log data (see `data/synthetic_trading_logs.py`) because no real trading log export was provided. Swap that module for a real loader against your actual trade blotter to run this on live data -- the DB schema, cleaning, feature engineering, and walk-forward model are all data-agnostic and require no changes.
