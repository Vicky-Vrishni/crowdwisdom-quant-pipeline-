CREATE TABLE IF NOT EXISTS macro_events (
    event_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    event_name          TEXT NOT NULL,
    event_ts_utc         TEXT NOT NULL,     
    event_ts_est         TEXT NOT NULL,      
    country              TEXT,
    forecast_value       REAL,
    actual_value         REAL,
    previous_value       REAL,
    surprise_score        REAL,               
    source                TEXT NOT NULL DEFAULT 'apify',
    scraped_at_utc         TEXT NOT NULL,
    UNIQUE(event_name, event_ts_utc)
);

CREATE TABLE IF NOT EXISTS trading_logs_raw (
    trade_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id            TEXT NOT NULL,
    ts_utc                TEXT NOT NULL,       
    direction             TEXT NOT NULL,        
    quantity              REAL NOT NULL,
    price                 REAL NOT NULL,
    pnl                   REAL NOT NULL,
    simulation_permutation  TEXT NOT NULL,
    raw_ingested_at_utc      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trading_logs_clean (
    trade_id             INTEGER PRIMARY KEY,
    account_id            TEXT NOT NULL,
    ts_utc                TEXT NOT NULL,
    direction             TEXT NOT NULL,
    quantity              REAL NOT NULL,
    price                 REAL NOT NULL,
    pnl                   REAL NOT NULL,
    simulation_permutation  TEXT NOT NULL,
    is_dedup_kept          INTEGER NOT NULL DEFAULT 1,
    dedup_group_id          TEXT,
    weekday               INTEGER,
    hour_of_day            INTEGER,
    minutes_since_last_macro_event REAL,
    nearest_macro_event      TEXT,
    nearest_macro_surprise    REAL,
    FOREIGN KEY(trade_id) REFERENCES trading_logs_raw(trade_id)
);

CREATE TABLE IF NOT EXISTS walk_forward_predictions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    fold_id               INTEGER NOT NULL,
    train_start           TEXT NOT NULL,
    train_end             TEXT NOT NULL,
    test_start            TEXT NOT NULL,
    test_end              TEXT NOT NULL,
    simulation_permutation  TEXT NOT NULL,
    hour_of_day            INTEGER NOT NULL,
    weekday               INTEGER NOT NULL,
    predicted_pnl          REAL,
    actual_pnl             REAL,
    n_trades              INTEGER
);

CREATE INDEX IF NOT EXISTS idx_macro_ts ON macro_events(event_ts_utc);
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trading_logs_raw(ts_utc);
CREATE INDEX IF NOT EXISTS idx_clean_ts ON trading_logs_clean(ts_utc);