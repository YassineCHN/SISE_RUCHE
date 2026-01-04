"""
Test de comptabilisation JobTeaser — LISTE ONLY
Objectif : estimer la volumétrie finale AVANT le gros scraping
Aucune page détail n'est ouverte.
"""

import time
from collections import defaultdict
from pprint import pprint

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
# PARAMÈTRES DE TEST (utilise la config finale)
# ============================================================

SORT_MODE = "recency"
PAGE_SLEEP = 2.0
CARD_SLEEP_EVERY = 15
CARD_SLEEP_TIME = 0.5


# ============================================================
# SCRIPT
# ============================================================


def main():
    print("🧪 TEST DE COMPTABILISATION — LISTE ONLY")
    print("=" * 70)
    print("🔧 CONFIG UTILISÉE")
    print(f"- SEARCH_QUERIES             : {config.SEARCH_QUERIES}")
    print(f"- MAX_LIST_PAGES_PER_KEYWORD : {config.MAX_LIST_PAGES_PER_KEYWORD}")
    print(f"- MAX_LIST_AGE_DAYS          : {config.MAX_LIST_AGE_DAYS}")
    print(f"- IGNORE_UNKNOWN_TIME        : {config.IGNORE_UNKNOWN_TIME}")
    print("=" * 70)

    driver = create_driver(headless=config.HEADLESS)
    driver.set_page_load_timeout(45)

    cookies_done = False
    seen_ids = set()

    global_stats = defaultdict(int)
    per_keyword = {}

    t0 = time.perf_counter()

    try:
        for query in config.SEARCH_QUERIES:
            print(f"\n🔎 Keyword : {query}")
            stats = defaultdict(int)
            page = 1

            while page <= config.MAX_LIST_PAGES_PER_KEYWORD:
                url = build_search_url(query, page=page, sort=SORT_MODE)

                try:
                    driver.get(url)
                except Exception as e:
                    print(f"⚠️ Erreur chargement page {page} : {e}")
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

                    # ✔️ ÉLIGIBLE DÉTAIL (POTENTIEL)
                    seen_ids.add(job["id"])
                    stats["eligible_detail"] += 1
                    global_stats["eligible_detail"] += 1

                page += 1
                time.sleep(PAGE_SLEEP)

            per_keyword[query] = dict(stats)

    finally:
        try:
            driver.quit()
        except Exception:
            pass

    elapsed = time.perf_counter() - t0

    # ========================================================
    # RÉSULTATS
    # ========================================================

    print("\n📊 STATS PAR KEYWORD")
    pprint(per_keyword)

    print("\n📊 STATS GLOBALES")
    pprint(dict(global_stats))

    total_cards = global_stats["cards_seen"] or 1
    eligible = global_stats["eligible_detail"]

    print("\n📉 FUNNEL GLOBAL")
    print("=" * 70)
    print(f"Cards vues                  : {total_cards}")
    print(f"Sans ID                     : {global_stats['no_id']}")
    print(f"Doublons globaux             : {global_stats['duplicates']}")
    print(f"Ignore unknown time         : {global_stats['ignored_unknown_time']}")
    print(f"Ignore trop anciennes       : {global_stats['ignored_too_old']}")
    print(f"Filtre titre data           : {global_stats['filtered_title']}")
    print("-" * 70)
    print(f"➡️ Éligibles ouverture détail : {eligible}")
    print(f"➡️ Taux de survie liste       : {eligible / total_cards * 100:.1f} %")

    # estimation réaliste
    est_low = int(eligible * 0.6)
    est_high = int(eligible * 0.8)

    print("\n📈 ESTIMATION VOLUME FINAL")
    print("=" * 70)
    print(f"- Potentiel liste (max)      : {eligible}")
    print(f"- Estimation réaliste finale : {est_low} → {est_high}")
    print(f"- Temps d'exécution          : {elapsed:.1f}s")

    print("\n✅ Test de comptabilisation terminé")


if __name__ == "__main__":
    main()
