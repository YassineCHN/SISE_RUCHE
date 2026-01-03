from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
import re
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from jobteaser.archives.utils_jt import (
    create_driver,
    accept_cookies,
    handle_cloudflare,
    build_search_url,
    get_job_cards,
    extract_job_preview,
)

# =========================
# PARAMÈTRES DU TEST #2
# =========================

SEARCH_QUERIES = [
    "data engineer",
    "data scientist",
    "intelligence artificielle",
    "machine learning",
    "data analyst",
    "consultant data",
]

MAX_LIST_AGE_DAYS = 30  # 🔥 plus strict
MAX_PAGES_PER_KEYWORD = 20  # sécurité
SLEEP_BETWEEN_PAGES = 2

# =========================
# UTILS TEMPS (LISTE)
# =========================


def parse_relative_time_fr(text):
    if not text:
        return None

    text = text.lower().strip()

    # cas immédiats
    if "aujourd" in text:
        return 0
    if "hier" in text:
        return 1

    # heures / minutes → considéré comme aujourd’hui
    if "heure" in text or "minute" in text:
        return 0

    # jours (singulier / pluriel)
    m = re.search(r"il y a (\d+) jour", text)
    if m:
        return int(m.group(1))

    # semaines
    m = re.search(r"il y a (\d+) semaine", text)
    if m:
        return int(m.group(1)) * 7

    # mois (approximation)
    m = re.search(r"il y a (\d+) mois", text)
    if m:
        return int(m.group(1)) * 30

    # fallback
    return None


def get_card_age_days(card, driver, timeout=5):
    try:
        time_elem = card.find_element(By.CSS_SELECTOR, "footer time")

        # attendre que le texte soit réellement injecté
        WebDriverWait(driver, timeout).until(
            lambda d: time_elem.get_attribute("innerText").strip() != ""
        )

        time_text = time_elem.get_attribute("innerText").strip()
        return parse_relative_time_fr(time_text)

    except:
        return None


def age_bucket(age):
    if age is None:
        print(age)
        return "unknown"
    if age <= 3:
        return "0-3j"
    if age <= 7:
        return "4-7j"
    if age <= 14:
        return "8-14j"
    return "15+j"


# =========================
# SCRIPT DE TEST
# =========================


def main():
    driver = create_driver(headless=False)

    seen_ids_global = set()

    global_stats = {
        "total_pages": 0,
        "total_cards": 0,
        "total_unique_candidates": 0,
    }

    per_query_stats = {}

    for query in SEARCH_QUERIES:
        print(f"\n🔎 TEST #2 — {query}")

        stats = {
            "pages": 0,
            "cards_seen": 0,
            "candidates_unique": 0,
            "duplicates_global": 0,
            "ignored_too_old": 0,
            "age_buckets": {
                "0-3j": 0,
                "4-7j": 0,
                "8-14j": 0,
                "15+j": 0,
                "unknown": 0,
            },
        }

        for page in range(1, MAX_PAGES_PER_KEYWORD + 1):
            page_url = build_search_url(query, page)
            driver.get(page_url)
            handle_cloudflare(driver)
            accept_cookies(driver)

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "[data-testid='jobad-card']")
                )
            )

            cards = get_job_cards(driver)

            stats["pages"] += 1
            global_stats["total_pages"] += 1

            print(f"  📄 Page {page} — {len(cards)} cards")

            for card in cards:
                stats["cards_seen"] += 1
                global_stats["total_cards"] += 1

                try:
                    job = extract_job_preview(card, driver)
                    if job is None:
                        continue
                except StaleElementReferenceException:
                    continue

                job_id = job["id"]

                if job_id in seen_ids_global:
                    stats["duplicates_global"] += 1
                    continue

                age_days = get_card_age_days(card, driver)

                if age_days is not None and age_days > MAX_LIST_AGE_DAYS:
                    stats["ignored_too_old"] += 1
                    continue

                # candidate unique
                seen_ids_global.add(job_id)
                stats["candidates_unique"] += 1
                global_stats["total_unique_candidates"] += 1

                stats["age_buckets"][age_bucket(age_days)] += 1

            time.sleep(SLEEP_BETWEEN_PAGES)

        per_query_stats[query] = stats

        dup_rate = (
            stats["duplicates_global"] / stats["cards_seen"] * 100
            if stats["cards_seen"]
            else 0
        )

        print(
            f"  ✅ pages: {stats['pages']} | "
            f"cards: {stats['cards_seen']} | "
            f"uniques: {stats['candidates_unique']} | "
            f"duplicates: {stats['duplicates_global']} ({dup_rate:.1f}%) | "
            f"ignorées âge: {stats['ignored_too_old']}"
        )
        print(f"     ⏱️ distribution âge: {stats['age_buckets']}")

    print("\n📊 RÉSUMÉ GLOBAL")
    print(f"- pages totales: {global_stats['total_pages']}")
    print(f"- cards vues: {global_stats['total_cards']}")
    print(f"- candidates uniques: {global_stats['total_unique_candidates']}")

    input("\nAppuie sur ENTER pour fermer le navigateur...")
    driver.quit()


if __name__ == "__main__":
    main()
