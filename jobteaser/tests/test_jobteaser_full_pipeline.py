import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
from datetime import datetime
from collections import defaultdict
from pathlib import Path

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from utils_jt import (
    create_driver,
    handle_cloudflare,
    build_search_url,
    get_job_cards,
    extract_job_preview,
    is_relevant_title,
)

from nlp_jobteaser import enrich_offers_jobteaser


# ============================================================
# CONFIG TEST
# ============================================================

KEYWORDS = [
    "data engineer",
    "data scientist",
    "data analyst",
    "machine learning",
    "intelligence artificielle",
]

MAX_DETAIL_PAGES_PER_KEYWORD = 10
MAX_DETAIL_PAGES_TOTAL = 50
MAX_LIST_PAGES_PER_KEYWORD = 5

OUTPUT_DIR = Path("output_test")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# TIMER UTILITY
# ============================================================


class Timer:
    def __init__(self):
        self.start_times = {}
        self.durations = defaultdict(float)

    def start(self, name):
        self.start_times[name] = time.perf_counter()

    def stop(self, name):
        if name in self.start_times:
            self.durations[name] += time.perf_counter() - self.start_times[name]

    def report(self):
        print("\n⏱️ TIMING REPORT")
        total = sum(self.durations.values())
        for k, v in sorted(self.durations.items(), key=lambda x: -x[1]):
            pct = (v / total * 100) if total else 0
            print(f"- {k:<30} {v:6.2f}s ({pct:4.1f}%)")
        print("-" * 40)
        print(f"TOTAL: {total:.2f}s\n")


# ============================================================
# SELENIUM HELPERS
# ============================================================


def safe_inner_text(driver, css, timeout=10):
    try:
        el = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, css))
        )
        WebDriverWait(driver, timeout).until(
            lambda d: (el.get_attribute("innerText") or "").strip() != ""
        )
        return (el.get_attribute("innerText") or "").strip()
    except:
        return None


def extract_description_raw(driver, timeout=10):
    try:
        container = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(
                (
                    By.CSS_SELECTOR,
                    "article[data-testid='jobad-DetailView__Description']",
                )
            )
        )
        return (container.get_attribute("innerText") or "").strip()
    except:
        return None


def extract_offer_detail(driver, url, keyword):
    driver.get(url)
    handle_cloudflare(driver)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "header"))
    )

    return {
        "source": "jobteaser",
        "url": url,
        "search_keyword": keyword,
        "scraped_at": datetime.now().isoformat(),
        "title": safe_inner_text(
            driver, "h1[data-testid='jobad-DetailView__Heading__title']"
        ),
        "company": safe_inner_text(
            driver, "[data-testid='jobad-DetailView__Heading__company_name']"
        ),
        "publication_date_raw": safe_inner_text(
            driver, "p[class*='PageHeader_publicationDate__']"
        ),
        "contract_raw": safe_inner_text(
            driver, "[data-testid*='CandidacyDetails__Contract']"
        ),
        "location_raw": safe_inner_text(
            driver, "[data-testid*='CandidacyDetails__Locations']"
        ),
        "salary_raw": safe_inner_text(
            driver, "[data-testid*='CandidacyDetails__Wage']"
        ),
        "remote_raw": safe_inner_text(
            driver, "[data-testid*='CandidacyDetails__Remote']"
        ),
        "start_date_raw": safe_inner_text(
            driver, "[data-testid*='CandidacyDetails__start_date']"
        ),
        "education_level_raw": safe_inner_text(
            driver, "[data-testid*='Summary__studyLevels'] dd"
        ),
        "function_raw": safe_inner_text(
            driver, "[data-testid*='Summary__function'] dd"
        ),
        "application_deadline": safe_inner_text(
            driver, "[data-testid*='Summary__application_deadline'] dd"
        ),
        "description_raw": extract_description_raw(driver),
    }


# ============================================================
# MAIN PIPELINE
# ============================================================


def main():
    timer = Timer()
    raw_offers = []
    seen_urls = set()

    # ---------- INIT ----------
    timer.start("init_driver")
    driver = create_driver(headless=False)
    handle_cloudflare(driver)
    timer.stop("init_driver")

    # ---------- LIST + DETAIL ----------
    total_detail_count = 0

    try:
        for keyword in KEYWORDS:
            if total_detail_count >= MAX_DETAIL_PAGES_TOTAL:
                break

            print(f"\n🔎 Keyword: {keyword}")
            selected_for_keyword = 0

            for page in range(1, MAX_LIST_PAGES_PER_KEYWORD + 1):
                if selected_for_keyword >= MAX_DETAIL_PAGES_PER_KEYWORD:
                    break
                if total_detail_count >= MAX_DETAIL_PAGES_TOTAL:
                    break

                timer.start("list_scraping")
                url = build_search_url(keyword, page=page)
                driver.get(url)
                handle_cloudflare(driver)

                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "[data-testid='jobad-card']")
                    )
                )
                cards = get_job_cards(driver)
                timer.stop("list_scraping")

                for card in cards:
                    if selected_for_keyword >= MAX_DETAIL_PAGES_PER_KEYWORD:
                        break
                    if total_detail_count >= MAX_DETAIL_PAGES_TOTAL:
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
                    selected_for_keyword += 1
                    total_detail_count += 1

            print(f"  ✔ {selected_for_keyword} offres retenues")

    finally:
        try:
            driver.quit()
        except:
            pass

    # ---------- WRITE RAW JSON ----------
    timer.start("write_json_raw")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = OUTPUT_DIR / f"jobteaser_raw_test_{ts}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_offers, f, ensure_ascii=False, indent=2)
    timer.stop("write_json_raw")

    # ---------- NLP ----------
    timer.start("nlp")
    enriched_offers = enrich_offers_jobteaser(raw_offers)
    timer.stop("nlp")

    avg_len = sum(o.get("content_length", 0) for o in enriched_offers) / max(
        1, len(enriched_offers)
    )
    print(f"- Avg content length (clean): {avg_len:.0f} chars")

    # ---------- WRITE ENRICHED JSON ----------
    timer.start("write_json_enriched")
    enriched_path = OUTPUT_DIR / f"jobteaser_enriched_test_{ts}.json"
    with open(enriched_path, "w", encoding="utf-8") as f:
        json.dump(enriched_offers, f, ensure_ascii=False, indent=2)
    timer.stop("write_json_enriched")

    # ---------- REPORT ----------
    print("\n📦 OUTPUT FILES")
    print(f"- RAW      : {raw_path}")
    print(f"- ENRICHED : {enriched_path}")

    timer.report()

    print(f"📊 STATS")
    print(f"- Offres brutes   : {len(raw_offers)}")
    print(f"- Offres enrichies: {len(enriched_offers)}")


if __name__ == "__main__":
    main()
