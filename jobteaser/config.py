# config.py

# =========================
# PARAMÈTRES SCRAPING
# =========================

SEARCH_QUERIES = [
    "data engineer",
    "data scientist",
    "data analyst",
    "intelligence artificielle",
    "machine learning",
    "consultant data",
]

MAX_LIST_AGE_DAYS = 30
IGNORE_UNKNOWN_TIME = True

MAX_DETAIL_PAGES_TOTAL = 2000
MAX_DETAIL_PAGES_PER_KEYWORD = 300
MAX_LIST_PAGES_PER_KEYWORD = 35

HEADLESS = False

# =========================
# OUTPUT
# =========================

OUTPUT_DIR = "output"
