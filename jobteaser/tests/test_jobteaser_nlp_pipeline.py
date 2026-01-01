import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# test_jobteaser_nlp_pipeline.py
from datetime import datetime
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from utils_jt import create_driver, handle_cloudflare
from nlp_jobteaser import enrich_offers_jobteaser


TEST_URLS = [
    "https://www.jobteaser.com/fr/job-offers/8e7a2580-c6f1-4d82-b81b-ad9f649ae68b-groupe-baudin-chateauneuf-technicien-production-h-f",
    "https://www.jobteaser.com/fr/job-offers/83cbab63-43b1-44f8-a959-b9db585c1430-sopra-steria-next-stage-consultant-e-transformation-digitale-en-aeronautique-toulouse",
]


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


def get_description_raw(driver, timeout=10):
    # même logique que tu utilises déjà côté scraper: bloc Description
    try:
        container = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(
                (
                    By.CSS_SELECTOR,
                    "article[data-testid='jobad-DetailView__Description']",
                )
            )
        )
        # innerText est plus fiable que .text en React
        txt = (container.get_attribute("innerText") or "").strip()
        return txt if txt else None
    except:
        return None


def extract_offer_raw(driver, url: str):
    driver.get(url)
    handle_cloudflare(driver)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "header"))
    )

    offer = {
        "source": "jobteaser",
        "url": url,
        "scraped_at": datetime.now().isoformat(),
        "title": safe_inner_text(
            driver, "h1[data-testid='jobad-DetailView__Heading__title']"
        )
        or None,
        "company": safe_inner_text(
            driver, "[data-testid='jobad-DetailView__Heading__company_name']"
        )
        or None,
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
        "description_raw": get_description_raw(driver),
    }
    return offer


def main():
    driver = create_driver(headless=False)

    raw_offers = []
    for url in TEST_URLS:
        print("\n🔗 Scrape brut:", url)
        raw_offers.append(extract_offer_raw(driver, url))

    driver.quit()

    print("\n🧠 NLP JobTeaser (offline)")
    enriched = enrich_offers_jobteaser(raw_offers)

    for o in enriched:
        print("\n==============================")
        print("Title:", o.get("title"))
        print("Role family:", o.get("role_family"))
        print("Experience years:", o.get("experience_years"))
        print("Languages:", o.get("languages"))
        print("Hard skills:", o.get("hard_skills"))
        print("Soft skills:", o.get("soft_skills"))
        print("content_length:", o.get("content_length"))


if __name__ == "__main__":
    main()
