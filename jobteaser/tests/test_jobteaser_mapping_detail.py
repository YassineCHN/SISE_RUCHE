import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils_jt import create_driver, handle_cloudflare
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime

TEST_URLS = [
    "https://www.jobteaser.com/fr/job-offers/8e7a2580-c6f1-4d82-b81b-ad9f649ae68b-groupe-baudin-chateauneuf-technicien-production-h-f",
    "https://www.jobteaser.com/fr/job-offers/83cbab63-43b1-44f8-a959-b9db585c1430-sopra-steria-next-stage-consultant-e-transformation-digitale-en-aeronautique-toulouse",
]


def safe_text_non_empty(driver, css, timeout=10):
    try:
        el = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, css))
        )
        WebDriverWait(driver, timeout).until(
            lambda d: (el.get_attribute("innerText") or "").strip() != ""
        )
        return el.get_attribute("innerText").strip()
    except:
        return None


def main():
    driver = create_driver(headless=False)

    for url in TEST_URLS:
        print("\n🔗", url)
        driver.get(url)
        handle_cloudflare(driver)

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "header"))
        )

        data = {
            # publication
            "publication_date_raw": safe_text_non_empty(
                driver, "p[class*='PageHeader_publicationDate__']"
            ),
            # contrat / lieu / salaire / remote
            "contract_raw": safe_text_non_empty(
                driver, "[data-testid*='CandidacyDetails__Contract']"
            ),
            "location_raw": safe_text_non_empty(
                driver, "[data-testid*='CandidacyDetails__Locations']"
            ),
            "salary_raw": safe_text_non_empty(
                driver, "[data-testid*='CandidacyDetails__Wage']"
            ),
            "remote_raw": safe_text_non_empty(
                driver, "[data-testid*='CandidacyDetails__Remote']"
            ),
            "start_date_raw": safe_text_non_empty(
                driver, "[data-testid*='CandidacyDetails__start_date']"
            ),
            # entreprise
            "company_type_raw": safe_text_non_empty(
                driver, "[data-testid*='company_businessType']"
            ),
            "company_size_raw": safe_text_non_empty(
                driver, "[data-testid*='company_size']"
            ),
            "company_sector_raw": safe_text_non_empty(
                driver, "[data-testid*='company_industry']"
            ),
            # summary
            "education_level_raw": safe_text_non_empty(
                driver, "[data-testid*='Summary__studyLevels'] dd"
            ),
            "function_raw": safe_text_non_empty(
                driver, "[data-testid*='Summary__function'] dd"
            ),
            "application_deadline": safe_text_non_empty(
                driver, "[data-testid*='Summary__application_deadline'] dd"
            ),
            "scraped_at": datetime.now().isoformat(),
        }

        print(data)

    input("\nENTER pour quitter")
    driver.quit()


if __name__ == "__main__":
    main()
