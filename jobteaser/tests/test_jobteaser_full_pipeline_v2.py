import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from jobteaser.archives.utils_jt import (
    create_driver,
    handle_cloudflare,
    build_search_url,
    get_job_cards,
    extract_job_preview,
    is_relevant_title,
    extract_offer_detail,
)

from jobteaser.archives.nlp_jobteaser import enrich_offers_jobteaser


# =====================
# CONFIG TEST
# =====================

KEYWORDS = [
    "data engineer",
    "data scientist",
    "data analyst",
    "machine learning",
    "intelligence artificielle",
]

MAX_LIST_PAGES_PER_KEYWORD = 5
MAX_DETAIL_PAGES_PER_KEYWORD = 10
MAX_DETAIL_PAGES_TOTAL = 50

OUTPUT_DIR = Path("output_test")
OUTPUT_DIR.mkdir(exist_ok=True)


# =====================
# TIMER
# =====================


class Timer:
    def __init__(self):
        self.t0 = {}
        self.durations = defaultdict(float)

    def start(self, key):
        self.t0[key] = time.perf_counter()

    def stop(self, key):
        self.durations[key] += time.perf_counter() - self.t0.get(key, 0)

    def report(self):
        print("\n⏱️ TIMING REPORT")
        total = sum(self.durations.values())
        for k, v in sorted(self.durations.items(), key=lambda x: -x[1]):
            pct = (v / total * 100) if total else 0
            print(f"- {k:<25} {v:6.2f}s ({pct:4.1f}%)")
        print("-" * 40)
        print(f"TOTAL: {total:.2f}s\n")


# =====================
# MAIN
# =====================


def main():
    timer = Timer()
    raw_offers = []
    seen_urls = set()
    total_detail = 0

    timer.start("init_driver")
    driver = create_driver(headless=False)
    handle_cloudflare(driver)
    timer.stop("init_driver")

    try:
        for keyword in KEYWORDS:
            if total_detail >= MAX_DETAIL_PAGES_TOTAL:
                break

            print(f"\n🔎 Keyword: {keyword}")
            kept_for_kw = 0

            for page in range(1, MAX_LIST_PAGES_PER_KEYWORD + 1):
                if kept_for_kw >= MAX_DETAIL_PAGES_PER_KEYWORD:
                    break
                if total_detail >= MAX_DETAIL_PAGES_TOTAL:
                    break

                timer.start("list_scraping")
                driver.get(build_search_url(keyword, page))
                handle_cloudflare(driver)

                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "[data-testid='jobad-card']")
                    )
                )
                cards = get_job_cards(driver)
                timer.stop("list_scraping")

                for card in cards:
                    if kept_for_kw >= MAX_DETAIL_PAGES_PER_KEYWORD:
                        break
                    if total_detail >= MAX_DETAIL_PAGES_TOTAL:
                        break

                    job = extract_job_preview(card, driver)
                    if not job or not job.get("url"):
                        continue
                    if job["url"] in seen_urls:
                        continue
                    if not is_relevant_title(job["title"]):
                        continue

                    seen_urls.add(job["url"])
                    print(f"  → détail: {job['title']}")

                    timer.start("detail_scraping")
                    offer = extract_offer_detail(driver, job["url"], keyword)
                    timer.stop("detail_scraping")

                    raw_offers.append(offer)
                    kept_for_kw += 1
                    total_detail += 1

            print(f"  ✔ {kept_for_kw} offres retenues")

    finally:
        try:
            driver.quit()
        except:
            pass

    # ===== JSON RAW =====
    timer.start("write_raw")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = OUTPUT_DIR / f"jobteaser_raw_test_{ts}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_offers, f, ensure_ascii=False, indent=2)
    timer.stop("write_raw")

    # ===== NLP =====
    timer.start("nlp")
    enriched = enrich_offers_jobteaser(raw_offers)
    timer.stop("nlp")

    avg_len = sum(o.get("content_length", 0) for o in enriched) / max(1, len(enriched))
    print(f"- Avg content length (clean): {avg_len:.0f} chars")

    # ===== JSON ENRICHED =====
    timer.start("write_enriched")
    enriched_path = OUTPUT_DIR / f"jobteaser_enriched_test_{ts}.json"
    with open(enriched_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    timer.stop("write_enriched")

    # ===== REPORT =====
    print("\n📦 OUTPUT FILES")
    print(f"- RAW      : {raw_path}")
    print(f"- ENRICHED : {enriched_path}")

    timer.report()
    print(f"📊 STATS")
    print(f"- Offres brutes   : {len(raw_offers)}")
    print(f"- Offres enrichies: {len(enriched)}")


if __name__ == "__main__":
    main()
