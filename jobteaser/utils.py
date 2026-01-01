# utils.py
# ============================================================
# Helpers Selenium + parsing JobTeaser
# Basé sur la logique/sélecteurs de utils_jt.py (référence)
# ============================================================

import time
import re
from datetime import datetime
from urllib.parse import urlencode

import undetected_chromedriver as uc

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException


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
# DRIVER
# =========================


def create_driver(headless: bool = False):
    options = uc.ChromeOptions()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    if headless:
        options.add_argument("--headless=new")

    driver = uc.Chrome(
        options=options,
        version_main=142,
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


def handle_cloudflare(driver, timeout=20):
    try:
        WebDriverWait(driver, timeout).until_not(
            EC.presence_of_element_located((By.XPATH, "//input[@type='checkbox']"))
        )
    except:
        pass


# =========================
# LISTE
# =========================


def get_job_cards(driver):
    return driver.find_elements(By.CSS_SELECTOR, "[data-testid='jobad-card']")


def get_non_empty_text(element, driver, timeout=5):
    WebDriverWait(driver, timeout).until(
        lambda d: element.get_attribute("innerText").strip() != ""
    )
    return element.get_attribute("innerText").strip()


def build_search_url(query: str, page: int = 1, sort: str = "recency") -> str:
    params = dict(FRANCE_LOCATION_PARAMS)
    params["q"] = query
    params["sort"] = sort
    params["page"] = str(page)
    return f"{BASE_SEARCH_URL}?{urlencode(params)}"


def extract_job_preview(card, driver):
    """
    VERSION REFERENCE (ne pas réécrire les sélecteurs)
    """
    job = {}
    try:
        link_elem = card.find_element(By.CSS_SELECTOR, "a[href*='/job-offers/']")
        job["url"] = link_elem.get_attribute("href")
    except StaleElementReferenceException:
        return None

    try:
        job["id"] = job["url"].split("/job-offers/")[1].split("-")[0]
    except Exception:
        return None

    job["title"] = "Titre non trouvé"
    try:
        title_elem = card.find_element(By.CSS_SELECTOR, "h3[class*='JobAdCard_title']")
        job["title"] = get_non_empty_text(title_elem, driver)
    except Exception:
        pass

    job["company"] = None

    try:
        job["location"] = card.find_element(
            By.CSS_SELECTOR, "[data-testid='jobad-card-location'] span"
        ).text.strip()
    except Exception:
        job["location"] = None

    try:
        job["contract"] = card.find_element(
            By.CSS_SELECTOR, "[data-testid='jobad-card-contract'] span"
        ).text.strip()
    except Exception:
        job["contract"] = None

    return job


# =========================
# AGE / TIME
# =========================


def parse_relative_time_fr(text: str):
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


def is_relevant_title(title: str) -> bool:
    if not title:
        return False
    t = title.lower()
    return any(k in t for k in DATA_KEYWORDS)


# =========================
# DETAIL
# =========================


def extract_offer_detail(driver, url, keyword=None):
    """
    Scrape une page détail JobTeaser et retourne un dict brut.
    """
    job_id = None
    match = re.search(r"/job-offers/([a-f0-9-]{36})", url)
    if match:
        job_id = match.group(1)
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
        "id": job_id,
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
