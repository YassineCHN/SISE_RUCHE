import time
from selenium.webdriver.common.by import By

from jobteaser.archives.utils_jt_selenium_old import (
    DriverPool,
    search_jobs_selenium,
    get_job_detail_selenium,
    RateLimiter,
)

# ============================
# CONFIG DEBUG
# ============================
KEYWORD = "data"
MAX_PAGES = 3
OFFERS_PER_PAGE = 1


# ============================
# MAIN DEBUG SCRIPT
# ============================
def main():
    print("\n=== DEBUG MODE : 3 OFFERS (1 PER PAGE) ===\n")

    driver_pool = DriverPool(pool_size=1, headless=False)
    search_driver = driver_pool.create_driver()
    limiter = RateLimiter(max_calls=1)

    seen_ids = set()
    debug_jobs = []

    try:
        # ============================
        # STEP 1 — SEARCH (CARDS)
        # ============================
        print("[STEP 1] Searching job cards\n")

        jobs = search_jobs_selenium(
            driver=search_driver,
            keyword=KEYWORD,
            seen_ids=seen_ids,
            max_pages=MAX_PAGES,
            test_mode=True,
        )

        # Keep only 1 job per page → max 3 jobs
        debug_jobs = jobs[:MAX_PAGES]

        print(f"\n[DEBUG] Jobs kept for detail scraping: {len(debug_jobs)}\n")

        for job in debug_jobs:
            job_id, title, company, location, contract, url = job

            print("\n==============================")
            print("[CARD DATA]")
            print(f"ID        : {job_id}")
            print(f"Title     : {title}")
            print(f"Company   : {company}")
            print(f"Location  : {location}")
            print(f"Contract  : {contract}")
            print(f"URL       : {url}")
            print("==============================\n")

        # ============================
        # STEP 2 — DETAIL SCRAPING
        # ============================
        print("\n[STEP 2] Scraping detail pages\n")

        for job in debug_jobs:
            job_id, title, company, location, contract, url = job
            driver = driver_pool.get_driver()

            try:
                limiter.wait()

                print("\n------------------------------")
                print("[DETAIL PAGE START]")
                print(f"Job ID : {job_id}")
                print(f"URL    : {url}")
                print("------------------------------")

                offer = get_job_detail_selenium(driver, job_id, url)

                if offer is None:
                    print("[RESULT] ❌ OFFER FILTERED OR FAILED")
                    continue

                print("\n[DETAIL PAGE RESULT]")
                print(f"Title        : {offer.intitule}")
                print(f"Entreprise   : {offer.entreprise}")
                print(f"Lieu         : {offer.lieu}")
                print(f"Contrat      : {offer.type_contrat}")
                print(f"Desc length  : {len(offer.description)}")
                print(f"Competences  : {offer.competences}")
                print(f"Keywords     : {offer.matched_keywords}")
                print("------------------------------")

            finally:
                driver_pool.return_driver(driver)

    finally:
        print("\n[CLEANUP]")
        try:
            search_driver.quit()
        except:
            pass
        driver_pool.close_all()

    print("\n=== DEBUG MODE FINISHED ===\n")


if __name__ == "__main__":
    main()
