"""
Test rapide du funnel JobTeaser (LISTE UNIQUEMENT, robuste)
Mesure les pertes à chaque étape SANS ouvrir les pages détail.
"""

import time
from collections import defaultdict
from pprint import pprint

from selenium.common.exceptions import TimeoutException, WebDriverException

import config
from utils import (
    create_driver,
    accept_cookies,
    handle_cloudflare,
    build_search_url,
    get_job_cards,
    extract_job_preview,
    get_card_age_days,
    is_relevant_title,
)


# ============================================================
# PARAMÈTRES DE TEST
# ============================================================

MAX_PAGES_PER_KEYWORD = 15  # suffisant pour stats
SORT_MODE = "recency"

PAGE_SLEEP = 2.0  # pause entre pages
CARD_SLEEP_EVERY = 15  # pause toutes les X cards
CARD_SLEEP_TIME = 0.5


# ============================================================
# TEST FUNNEL
# ============================================================


def main():
    driver = create_driver(headless=config.HEADLESS)
    driver.set_page_load_timeout(45)

    cookies_done = False
    seen_ids = set()

    global_stats = defaultdict(int)
    per_keyword = {}

    print("🧪 TEST FUNNEL LISTE JOBTEASER (ROBUSTE)")
    print("=" * 60)

    try:
        for query in config.SEARCH_QUERIES:
            stats = defaultdict(int)
            page = 1

            print(f"\n🔎 Keyword: {query}")

            while page <= MAX_PAGES_PER_KEYWORD:
                url = build_search_url(query, page=page, sort=SORT_MODE)

                try:
                    driver.get(url)
                except TimeoutException:
                    print(f"⚠️ Timeout page {page} — skip")
                    break
                except WebDriverException as e:
                    print(f"❌ WebDriver error — skip keyword: {e}")
                    break

                handle_cloudflare(driver)

                if not cookies_done:
                    accept_cookies(driver)
                    cookies_done = True

                cards = get_job_cards(driver)
                if not cards:
                    break

                for i, card in enumerate(cards, 1):
                    stats["cards_seen"] += 1
                    global_stats["cards_seen"] += 1

                    if i % CARD_SLEEP_EVERY == 0:
                        time.sleep(CARD_SLEEP_TIME)

                    job = extract_job_preview(card, driver)
                    if not job or not job.get("id"):
                        stats["no_id"] += 1
                        global_stats["no_id"] += 1
                        continue

                    if job["id"] in seen_ids:
                        stats["duplicates"] += 1
                        global_stats["duplicates"] += 1
                        continue

                    age_days = get_card_age_days(card, driver)
                    if age_days is None:
                        if config.IGNORE_UNKNOWN_TIME:
                            stats["ignored_unknown_time"] += 1
                            global_stats["ignored_unknown_time"] += 1
                            continue
                    else:
                        if age_days > config.MAX_LIST_AGE_DAYS:
                            stats["ignored_too_old"] += 1
                            global_stats["ignored_too_old"] += 1
                            continue

                    if not is_relevant_title(job.get("title", "")):
                        stats["filtered_title"] += 1
                        global_stats["filtered_title"] += 1
                        continue

                    seen_ids.add(job["id"])
                    stats["eligible_for_detail"] += 1
                    global_stats["eligible_for_detail"] += 1

                page += 1
                time.sleep(PAGE_SLEEP)

            per_keyword[query] = dict(stats)

    finally:
        try:
            driver.quit()
        except Exception:
            pass

    # ========================================================
    # RÉSULTATS
    # ========================================================

    print("\n📊 STATS PAR KEYWORD")
    pprint(per_keyword)

    print("\n📊 STATS GLOBALES")
    pprint(dict(global_stats))

    total = global_stats["cards_seen"] or 1
    eligible = global_stats["eligible_for_detail"]

    print("\n📉 FUNNEL GLOBAL")
    print("=" * 60)
    print(f"Cards vues                  : {total}")
    print(f"Sans ID                     : {global_stats['no_id']}")
    print(f"Doublons                    : {global_stats['duplicates']}")
    print(f"Ignore unknown time         : {global_stats['ignored_unknown_time']}")
    print(f"Ignore trop anciennes       : {global_stats['ignored_too_old']}")
    print(f"Filtre titre data           : {global_stats['filtered_title']}")
    print("-" * 60)
    print(f"➡️ Éligibles ouverture détail : {eligible}")
    print(f"➡️ Taux de survie            : {eligible / total * 100:.1f} %")

    print("\n✅ Test funnel terminé")


if __name__ == "__main__":
    main()
