import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "db" / "cwt_pipeline.sqlite"
OUTPUT_DIR = BASE_DIR / "output" / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "db").mkdir(parents=True, exist_ok=True)

APIFY_API_TOKEN = os.environ.get("APIFY_API_TOKEN", "")

APIFY_MACRO_CALENDAR_ACTOR_ID = os.environ.get(
    "APIFY_MACRO_CALENDAR_ACTOR_ID", "epctex/investing-calendar-scraper"
)
ALLOW_SYNTHETIC_FALLBACK = True
TRAIN_WINDOW_DAYS = 30
TEST_WINDOW_DAYS = 7
STEP_DAYS = 7  # how far the window slides forward each iteration
LOOKBACK_DAYS = 180
N_ACCOUNTS = 5
SIMULATION_PERMUTATIONS = [
    "A_scalp_5m", "B_trend_15m", "C_meanrev_1h", "D_breakout_30m", "E_news_fade_5m"
]

RANDOM_SEED = 42