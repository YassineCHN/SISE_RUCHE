import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from jobteaser.archives.utils_jt import (
    create_driver,
    handle_cloudflare,
    build_search_url,
    get_job_cards,
    extract_job_preview,
    is_relevant_title,
)
from jobteaser.archives.nlp_jobteaser import enrich_offers_jobteaser


SEARCH_QUERIES = [
    "data engineer",
    "data scientist",
]

OFFERS_PER_QUERY = 4
MAX_PAGES_PER_QUERY = 5


# ---------------------------
# Helpers Selenium
# ---------------------------


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


def extract_offer_raw(driver, url, search_query):
    driver.get(url)
    handle_cloudflare(driver)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "header"))
    )

    return {
        "source": "jobteaser",
        "url": url,
        "search_keyword": search_query,
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
        "description_raw": get_description_raw(driver),
    }


# ---------------------------
# Main mini-run
# ---------------------------


def main():
    driver = create_driver(headless=False)
    raw_offers = []

    try:
        for query in SEARCH_QUERIES:
            print(f"\n🔎 Recherche JobTeaser : {query}")
            selected = 0

            for page in range(1, MAX_PAGES_PER_QUERY + 1):
                if selected >= OFFERS_PER_QUERY:
                    break

                page_url = build_search_url(query, page=page)
                driver.get(page_url)
                handle_cloudflare(driver)

                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "[data-testid='jobad-card']")
                    )
                )

                cards = get_job_cards(driver)

                for card in cards:
                    if selected >= OFFERS_PER_QUERY:
                        break

                    job = extract_job_preview(card, driver)
                    if not job or not job.get("url"):
                        continue

                    if not is_relevant_title(job["title"]):
                        continue

                    print(f"  → sélection (data): {job['title']}")
                    offer = extract_offer_raw(driver, job["url"], query)
                    raw_offers.append(offer)
                    selected += 1

            print(f"  ✔ {selected} offres sélectionnées pour '{query}'")

        print("\n🧠 NLP JobTeaser (offline)")
        enriched = enrich_offers_jobteaser(raw_offers)

        for o in enriched:
            print("\n==============================")
            print("Keyword:", o.get("search_keyword"))
            print("Title:", o.get("title"))
            print("Role family:", o.get("role_family"))
            print("Experience years:", o.get("experience_years"))
            print("Languages:", o.get("languages"))
            print("Hard skills:", o.get("hard_skills"))
            print("Soft skills:", o.get("soft_skills"))
            print("content_length:", o.get("content_length"))

    finally:
        try:
            driver.quit()
        except:
            pass


if __name__ == "__main__":
    main()
