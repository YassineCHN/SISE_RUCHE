from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
import undetected_chromedriver as uc
import time
import json
import os
from urllib.parse import urljoin


JOBTEASER_URL = "https://www.jobteaser.com/fr/job-offers"


def create_driver(headless=False):
    """
    Crée un driver Chrome furtif (Cloudflare-friendly)
    """
    options = uc.ChromeOptions()

    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")

    if headless:
        options.add_argument("--headless=new")

    driver = uc.Chrome(
        options=options,
        version_main=142,  # ⚠️ adapte si ta version Chrome est différente
    )

    driver.set_page_load_timeout(30)
    driver.implicitly_wait(5)

    return driver


def accept_cookies(driver):
    """
    Clique sur le bouton 'Accepter' des cookies s'il apparaît
    """
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
        print("✅ Cookies acceptés")
    except:
        print("ℹ️ Pas de bannière cookies détectée")


def get_non_empty_text(element, driver, timeout=5):
    WebDriverWait(driver, timeout).until(
        lambda d: element.get_attribute("innerText").strip() != ""
    )
    return element.get_attribute("innerText").strip()


def get_job_description(driver, timeout=10):
    """
    Extrait la description complète d'une offre JobTeaser
    """
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

        # attendre que le texte soit injecté
        WebDriverWait(driver, timeout).until(
            lambda d: content.get_attribute("innerText").strip() != ""
        )

        return content.get_attribute("innerText").strip()

    except Exception as e:
        return "Description non trouvée"


def handle_cloudflare(driver, timeout=20):
    """
    Attend que le challenge Cloudflare disparaisse
    """
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
        return None  # carte morte, on l’ignore

    # ID
    job["id"] = job["url"].split("/job-offers/")[1].split("-")[0]

    # Titre
    job["title"] = "Titre non trouvé"
    try:
        title_elem = card.find_element(By.CSS_SELECTOR, "h3[class*='JobAdCard_title']")
        job["title"] = get_non_empty_text(title_elem, driver)
    except:
        pass

    # Entreprise
    job["company"] = None

    # Localisation
    try:
        job["location"] = card.find_element(
            By.CSS_SELECTOR, "[data-testid='jobad-card-location'] span"
        ).text.strip()
    except:
        job["location"] = None

    # Contrat
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

        except Exception as e:
            print(f"⚠️ Détail échoué ({attempt+1}) : {job['title']}")

    job["description"] = None
    return job


def save_to_json(data, filename):
    os.makedirs("json_scrapped", exist_ok=True)

    path = os.path.join("json_scrapped", filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n💾 JSON sauvegardé : {path}")


def main():
    driver = create_driver(headless=False)

    results = []
    failed_jobs = []
    MAX_PAGES = 3
    MAX_PER_PAGE = 2
    seen_ids = set()

    print("🌍 Démarrage du scraping JobTeaser")

    for page in range(1, MAX_PAGES + 1):
        print(f"\n📄 PAGE {page}")
        page_url = f"{JOBTEASER_URL}?page={page}"
        driver.get(page_url)
        handle_cloudflare(driver)
        accept_cookies(driver)

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "[data-testid='jobad-card']")
            )
        )

        cards = get_job_cards(driver)
        print(f"→ {len(cards)} offres détectées")

        for card in cards[:MAX_PER_PAGE]:
            job = extract_job_preview(card, driver)
            if job is None:
                print("  ⚠️ Carte ignorée (stale element)")
                continue
            if job["id"] in seen_ids:
                print("  ↩️ Offre déjà vue, ignorée")
                continue
            seen_ids.add(job["id"])

            print(f"  • {job['title']}")

            job = enrich_with_detail(driver, job)
            if job["description"] is None:
                failed_jobs.append(job["url"])
            else:
                results.append(job)

    save_to_json(results, "resultats_test.json")

    print(f"\n✅ Scraping terminé : {len(results)} offres collectées")
    print(f"\n❌ Offres échouées : {len(failed_jobs)}")
    input("\nAppuie sur ENTER pour fermer le navigateur...")
    driver.quit()


if __name__ == "__main__":
    main()
