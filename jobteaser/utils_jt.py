from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
import undetected_chromedriver as uc
import time
import json
import os
import re
from urllib.parse import urlencode
from datetime import datetime


BASE_SEARCH_URL = "https://www.jobteaser.com/fr/job-offers"
DATA_KEYWORDS = [
    "data",
    "engineer",
    "scientist",
    "analytics",
    "analyst",
    "machine learning",
    "ml",
    "ai",
    "business intelligence",
    "bi",
]

FRANCE_LOCATION_PARAMS = {
    "lat": "46.711046499999995",
    "lng": "2.1811786692949857",
    "localized_location": "France",
    "location": "France::_Y291bnRyeTo6OnVGaW9mQWV3VEVWbzlSc056bVZmZU5jOEFyTT0=",
}


# =========================
# PARAMÈTRES "FINAL"
# =========================
MAX_LIST_AGE_DAYS = 30
IGNORE_UNKNOWN_TIME = True  # offres "de la semaine" / sponsorisées
MAX_DETAIL_PAGES_TOTAL = 1500
MAX_DETAIL_PAGES_PER_KEYWORD = 250

SEARCH_QUERIES = [
    "data analyst",
    "data engineer",
    "data scientist",
    "intelligence artificielle",
    "machine learning",
    "consultant data",
]


def is_relevant_title(title):
    title = title.lower()
    return any(keyword in title for keyword in DATA_KEYWORDS)


def create_driver(headless=False):
    options = uc.ChromeOptions()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")

    if headless:
        options.add_argument("--headless=new")

    driver = uc.Chrome(
        options=options,
        version_main=142,  # adapte si ta version Chrome est différente
    )

    driver.set_page_load_timeout(30)
    driver.implicitly_wait(5)
    return driver


def accept_cookies(driver):
    """Clique sur le bouton cookies s'il apparaît (silencieux si absent)."""
    try:
        button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "//button[contains(text(), 'Accepter') or contains(text(), 'Accept')]",
                )
            )
        )
        button.click()
    except:
        pass


def get_non_empty_text(element, driver, timeout=5):
    WebDriverWait(driver, timeout).until(
        lambda d: element.get_attribute("innerText").strip() != ""
    )
    return element.get_attribute("innerText").strip()


def get_job_description(driver, timeout=10):
    try:
        container = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(
                (
                    By.CSS_SELECTOR,
                    "article[data-testid='jobad-DetailView__Description']",
                )
            )
        )
        content = container.find_element(
            By.CSS_SELECTOR, "div[class*='Description_content']"
        )
        WebDriverWait(driver, timeout).until(
            lambda d: content.get_attribute("innerText").strip() != ""
        )
        return content.get_attribute("innerText").strip()
    except:
        return "Description non trouvée"


def handle_cloudflare(driver, timeout=20):
    try:
        WebDriverWait(driver, timeout).until_not(
            EC.presence_of_element_located((By.XPATH, "//input[@type='checkbox']"))
        )
    except:
        pass


def get_job_cards(driver):
    return driver.find_elements(By.CSS_SELECTOR, "[data-testid='jobad-card']")


def extract_job_preview(card, driver):
    job = {}
    try:
        link_elem = card.find_element(By.CSS_SELECTOR, "a[href*='/job-offers/']")
        job["url"] = link_elem.get_attribute("href")
    except StaleElementReferenceException:
        return None

    job["id"] = job["url"].split("/job-offers/")[1].split("-")[0]

    job["title"] = "Titre non trouvé"
    try:
        title_elem = card.find_element(By.CSS_SELECTOR, "h3[class*='JobAdCard_title']")
        job["title"] = get_non_empty_text(title_elem, driver)
    except:
        pass

    job["company"] = None

    try:
        job["location"] = card.find_element(
            By.CSS_SELECTOR, "[data-testid='jobad-card-location'] span"
        ).text.strip()
    except:
        job["location"] = None

    try:
        job["contract"] = card.find_element(
            By.CSS_SELECTOR, "[data-testid='jobad-card-contract'] span"
        ).text.strip()
    except:
        job["contract"] = None

    return job


def get_company_name(driver, timeout=5):
    try:
        elem = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(
                (
                    By.CSS_SELECTOR,
                    "[data-testid='jobad-DetailView__Heading__company_name']",
                )
            )
        )
        return elem.text.strip()
    except:
        return None


def enrich_with_detail(driver, job, retries=1):
    for attempt in range(retries + 1):
        try:
            driver.get(job["url"])
            handle_cloudflare(driver)

            job["company"] = get_company_name(driver)
            job["description"] = get_job_description(driver)

            if job["description"] != "Description non trouvée":
                return job
        except:
            print(f"⚠️ Détail échoué ({attempt+1}) : {job.get('title')}")

    job["description"] = None
    return job


def get_run_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_results_and_stats(results, stats):
    os.makedirs("json_scrapped", exist_ok=True)
    ts = get_run_timestamp()

    results_path = f"json_scrapped/jobs_{ts}.json"
    stats_path = f"json_scrapped/stats_{ts}.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Résultats sauvegardés : {results_path}")
    print(f"📊 Statistiques sauvegardées : {stats_path}")


def build_search_url(query, page):
    params = {**FRANCE_LOCATION_PARAMS, "q": query, "page": page}
    return f"{BASE_SEARCH_URL}?{urlencode(params)}"


# =========================
# TIME LISTE (pré-filtre)
# =========================


def parse_relative_time_fr(text):
    if not text:
        return None

    text = text.lower().strip()

    if "aujourd" in text:
        return 0
    if "hier" in text:
        return 1
    if "heure" in text or "minute" in text:
        return 0

    m = re.search(r"il y a (\d+) jour", text)
    if m:
        return int(m.group(1))

    m = re.search(r"il y a (\d+) semaine", text)
    if m:
        return int(m.group(1)) * 7

    m = re.search(r"il y a (\d+) mois", text)
    if m:
        return int(m.group(1)) * 30

    return None


def get_card_age_days(card, driver, timeout=5):
    """
    Récupère l'âge (jours) affiché sur la card via <footer><time>...</time></footer>.
    Utilise innerText + wait (React hydration).
    Retourne None si inconnu (offres "de la semaine", sponsorisées, etc.).
    """
    try:
        time_elem = card.find_element(By.CSS_SELECTOR, "footer time")
        WebDriverWait(driver, timeout).until(
            lambda d: time_elem.get_attribute("innerText").strip() != ""
        )
        txt = time_elem.get_attribute("innerText").strip()
        return parse_relative_time_fr(txt)
    except:
        return None


def extract_offer_detail(driver, url, keyword=None):
    """
    Scrape une page détail JobTeaser et retourne un dict brut.
    """
    driver.get(url)
    handle_cloudflare(driver)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "header"))
    )

    def safe_text(css):
        try:
            el = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, css))
            )
            return (el.get_attribute("innerText") or "").strip()
        except:
            return None

    def safe_desc():
        try:
            container = WebDriverWait(driver, 10).until(
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

    return {
        "source": "jobteaser",
        "url": url,
        "search_keyword": keyword,
        "scraped_at": datetime.now().isoformat(),
        "title": safe_text("h1[data-testid='jobad-DetailView__Heading__title']"),
        "company": safe_text("[data-testid='jobad-DetailView__Heading__company_name']"),
        "publication_date_raw": safe_text("p[class*='PageHeader_publicationDate__']"),
        "contract_raw": safe_text("[data-testid*='CandidacyDetails__Contract']"),
        "location_raw": safe_text("[data-testid*='CandidacyDetails__Locations']"),
        "salary_raw": safe_text("[data-testid*='CandidacyDetails__Wage']"),
        "remote_raw": safe_text("[data-testid*='CandidacyDetails__Remote']"),
        "start_date_raw": safe_text("[data-testid*='CandidacyDetails__start_date']"),
        "education_level_raw": safe_text("[data-testid*='Summary__studyLevels'] dd"),
        "function_raw": safe_text("[data-testid*='Summary__function'] dd"),
        "application_deadline": safe_text(
            "[data-testid*='Summary__application_deadline'] dd"
        ),
        "description_raw": safe_desc(),
    }


def clean_job_description(text: str) -> str:
    """
    Nettoyage robuste des descriptions JobTeaser.
    Objectif : une seule ligne propre pour NLP / stockage.
    """
    if not text:
        return ""

    # normaliser
    text = text.replace("\r", " ").replace("\xa0", " ")

    # supprimer sauts de ligne multiples
    text = re.sub(r"\n+", " ", text)

    # espaces multiples
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================
# MAIN
# =========================


def main():
    driver = create_driver(headless=False)

    results = []
    failed_jobs = []

    # dédoublonnage global
    seen_ids = set()

    # limites
    total_detail_opened = 0

    print("🌍 Démarrage du scraping JobTeaser")

    stats = {}
    cookies_done = False

    for query in SEARCH_QUERIES:
        print(f"\n🔎 Recherche : {query}")
        stats[query] = {
            "cards_seen": 0,
            "ignored_unknown_time": 0,
            "ignored_too_old": 0,
            "duplicates": 0,
            "filtered_title": 0,
            "detail_opened": 0,
            "kept": 0,
            "failed": 0,
        }

        page = 1
        while True:
            # stop global dur
            if total_detail_opened >= MAX_DETAIL_PAGES_TOTAL:
                print("\n⛔ Limite globale pages détail atteinte. Arrêt.")
                break

            # stop par mot-clé dur
            if stats[query]["detail_opened"] >= MAX_DETAIL_PAGES_PER_KEYWORD:
                print(
                    "  ⛔ Limite pages détail par mot-clé atteinte. Passage au suivant."
                )
                break

            page_url = build_search_url(query, page)
            driver.get(page_url)
            handle_cloudflare(driver)

            if not cookies_done:
                accept_cookies(driver)
                cookies_done = True

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "[data-testid='jobad-card']")
                )
            )

            cards = get_job_cards(driver)
            if not cards:
                print("  ℹ️ Plus de résultats (0 card).")
                break

            print(f"\n📄 PAGE {page} — '{query}' → {len(cards)} cards")

            for card in cards:
                stats[query]["cards_seen"] += 1

                job = extract_job_preview(card, driver)
                if job is None:
                    continue

                if job["id"] in seen_ids:
                    stats[query]["duplicates"] += 1
                    continue

                # pré-filtre âge (liste)
                age_days = get_card_age_days(card, driver)
                if age_days is None:
                    if IGNORE_UNKNOWN_TIME:
                        stats[query]["ignored_unknown_time"] += 1
                        continue
                else:
                    if age_days > MAX_LIST_AGE_DAYS:
                        stats[query]["ignored_too_old"] += 1
                        continue

                # filtre titre (optionnel, tu l'avais déjà)
                if not is_relevant_title(job["title"]):
                    stats[query]["filtered_title"] += 1
                    continue

                # OK → on va en détail
                if total_detail_opened >= MAX_DETAIL_PAGES_TOTAL:
                    break
                if stats[query]["detail_opened"] >= MAX_DETAIL_PAGES_PER_KEYWORD:
                    break

                seen_ids.add(job["id"])
                total_detail_opened += 1
                stats[query]["detail_opened"] += 1

                print(f"  • {job['title']}")

                job = enrich_with_detail(driver, job)

                if job["description"] is None:
                    failed_jobs.append(job["url"])
                    stats[query]["failed"] += 1
                else:
                    results.append(job)
                    stats[query]["kept"] += 1

            # pagination suivante
            page += 1
            time.sleep(1.5)

        # arrêt global : sortir de la boucle query aussi
        if total_detail_opened >= MAX_DETAIL_PAGES_TOTAL:
            break

    # résumé
    print("\n📊 STATISTIQUES PAR REQUÊTE")
    for query, s in stats.items():
        print(
            f"- {query:28} | "
            f"cards: {s['cards_seen']:4} | "
            f"detail: {s['detail_opened']:4} | "
            f"kept: {s['kept']:4} | "
            f"dup: {s['duplicates']:4} | "
            f"old: {s['ignored_too_old']:4} | "
            f"unknown: {s['ignored_unknown_time']:4} | "
            f"filtered: {s['filtered_title']:4} | "
            f"failed: {s['failed']:3}"
        )

    save_results_and_stats(results, stats)

    print(f"\n✅ Scraping terminé : {len(results)} offres collectées")
    print(f"🔎 Pages détail ouvertes : {total_detail_opened}")
    print(f"❌ Offres échouées : {len(failed_jobs)}")

    input("\nAppuie sur ENTER pour fermer le navigateur...")
    driver.quit()


if __name__ == "__main__":
    main()
