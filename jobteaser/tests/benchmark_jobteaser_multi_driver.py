import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, get_context

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from utils_jt import (
    create_driver,
    handle_cloudflare,
    build_search_url,
    get_job_cards,
    extract_job_preview,
    is_relevant_title,
)

# ============================================================
# CONFIG
# ============================================================

KEYWORDS = [
    "data engineer",
    "data scientist",
]

MAX_URLS_TOTAL = 20
MAX_LIST_PAGES = 5
PROCESS_COUNTS = [1, 2, 3]  # 🔥 multiprocessing réaliste sous Windows

OUTPUT_DIR = Path("output_benchmark")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LIST SCRAPING (SÉQUENTIEL)
# ============================================================


def wait_for_cards(driver, timeout=12):
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "[data-testid='jobad-card']")
            )
        )
        return True
    except TimeoutException:
        return False


def collect_urls():
    print("🔎 Collecte des URLs (séquentielle)")
    urls = []
    seen = set()

    driver = create_driver(headless=False)
    try:
        for keyword in KEYWORDS:
            print(f"  · keyword: {keyword}")
            for page in range(1, MAX_LIST_PAGES + 1):
                if len(urls) >= MAX_URLS_TOTAL:
                    break

                driver.get(build_search_url(keyword, page=page))
                handle_cloudflare(driver)

                if not wait_for_cards(driver):
                    print(f"    ⚠️ page {page}: cards non détectées")
                    continue

                for card in get_job_cards(driver):
                    if len(urls) >= MAX_URLS_TOTAL:
                        break

                    job = extract_job_preview(card, driver)
                    if not job or not job.get("url"):
                        continue
                    if job["url"] in seen:
                        continue
                    if not is_relevant_title(job["title"]):
                        continue

                    seen.add(job["url"])
                    urls.append(job["url"])

        print(f"✔ {len(urls)} URLs collectées")
        return urls

    finally:
        try:
            driver.quit()
        except:
            pass


# ============================================================
# DETAIL SCRAPING (MULTIPROCESS)
# ============================================================


def scrape_detail_process(url):
    """
    ⚠️ Fonction exécutée dans un PROCESS.
    DOIT être définie au top-level (pickle).
    """
    driver = create_driver(headless=False)
    try:
        driver.get(url)
        handle_cloudflare(driver)

        WebDriverWait(driver, 12).until(
            EC.presence_of_element_located((By.TAG_NAME, "header"))
        )

        title = driver.find_element(
            By.CSS_SELECTOR, "h1[data-testid='jobad-DetailView__Heading__title']"
        ).text.strip()

        return {
            "url": url,
            "title": title,
        }

    except Exception as e:
        return {
            "url": url,
            "error": str(e),
        }

    finally:
        try:
            driver.quit()
        except:
            pass


# ============================================================
# BENCHMARK
# ============================================================


def run_benchmark(urls, n_processes):
    print(f"\n🚀 Benchmark multiprocessing ({n_processes} process)")
    start = time.perf_counter()

    ctx = get_context("spawn")  # 🔥 indispensable sous Windows
    with ctx.Pool(processes=n_processes) as pool:
        results = pool.map(scrape_detail_process, urls)

    duration = time.perf_counter() - start
    success = sum(1 for r in results if "title" in r)

    return {
        "processes": n_processes,
        "pages_requested": len(urls),
        "pages_success": success,
        "total_time": duration,
        "time_per_page": duration / max(1, success),
        "pages_per_second": success / duration if duration else 0,
    }


# ============================================================
# MAIN
# ============================================================


def main():
    urls = collect_urls()

    all_results = []
    for n in PROCESS_COUNTS:
        stats = run_benchmark(urls, n)
        all_results.append(stats)

        print(
            f"  - {n} proc | "
            f"{stats['pages_success']}/{stats['pages_requested']} pages | "
            f"{stats['total_time']:.1f}s | "
            f"{stats['time_per_page']:.2f}s/page"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / f"benchmark_multiprocessing_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n📊 RÉSUMÉ FINAL")
    for r in all_results:
        print(
            f"- {r['processes']} process | "
            f"{r['pages_success']} pages | "
            f"{r['total_time']:.1f}s | "
            f"{r['pages_per_second']:.2f} pages/s"
        )

    print(f"\n📁 Résultats sauvegardés → {out}")


if __name__ == "__main__":
    main()
