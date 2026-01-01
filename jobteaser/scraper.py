# scraper.py

from utils import (
    create_driver,
    accept_cookies,
    handle_cloudflare,
    build_search_url,
    get_job_cards,
    extract_job_preview,
    get_card_age_days,
    is_relevant_title,
    extract_offer_detail,
)
import config


def run_scraper():
    driver = create_driver(headless=config.HEADLESS)

    results = []
    failed = []

    seen_ids = set()
    cookies_done = False

    stats = {}
    total_detail_opened = 0

    print("🌍 Démarrage du scraping JobTeaser")

    try:
        for query in config.SEARCH_QUERIES:
            stats[query] = {
                "cards_seen": 0,
                "duplicates": 0,
                "ignored_unknown_time": 0,
                "ignored_too_old": 0,
                "filtered_title": 0,
                "detail_opened": 0,
                "kept": 0,
                "failed": 0,
            }

            page = 1
            while True:
                if total_detail_opened >= config.MAX_DETAIL_PAGES_TOTAL:
                    break
                if stats[query]["detail_opened"] >= config.MAX_DETAIL_PAGES_PER_KEYWORD:
                    break
                if page > getattr(config, "MAX_LIST_PAGES_PER_KEYWORD", 20):
                    break

                driver.get(build_search_url(query, page=page, sort="recency"))
                handle_cloudflare(driver)

                if not cookies_done:
                    accept_cookies(driver)
                    cookies_done = True

                cards = get_job_cards(driver)
                if not cards:
                    break

                for card in cards:
                    stats[query]["cards_seen"] += 1

                    job = extract_job_preview(card, driver)
                    if not job or not job.get("id") or not job.get("url"):
                        continue

                    # dédoublonnage global
                    if job["id"] in seen_ids:
                        stats[query]["duplicates"] += 1
                        continue

                    # age (None => offres de la semaine)
                    age_days = get_card_age_days(card, driver)
                    if age_days is None:
                        if config.IGNORE_UNKNOWN_TIME:
                            stats[query]["ignored_unknown_time"] += 1
                            continue
                    else:
                        if age_days > config.MAX_LIST_AGE_DAYS:
                            stats[query]["ignored_too_old"] += 1
                            continue

                    # filtre titre data
                    if not is_relevant_title(job.get("title", "")):
                        stats[query]["filtered_title"] += 1
                        continue

                    # on décide d'ouvrir le détail
                    seen_ids.add(job["id"])
                    stats[query]["detail_opened"] += 1
                    total_detail_opened += 1

                    offer = extract_offer_detail(driver, job["url"], keyword=query)

                    if offer.get("description_raw"):
                        results.append(offer)
                        stats[query]["kept"] += 1
                    else:
                        failed.append(job["url"])
                        stats[query]["failed"] += 1

                    if total_detail_opened >= config.MAX_DETAIL_PAGES_TOTAL:
                        break
                    if (
                        stats[query]["detail_opened"]
                        >= config.MAX_DETAIL_PAGES_PER_KEYWORD
                    ):
                        break

                page += 1

        return results, stats

    finally:
        try:
            driver.quit()
        except Exception:
            pass
