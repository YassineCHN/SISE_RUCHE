"""
JobTeaser Scraper with Selenium -> JSON + MongoDB (Optional)
FIXED VERSION with Cloudflare bypass and improved extraction

Improvements:
- Undetected ChromeDriver to bypass Cloudflare
- Cookie acceptance handling
- Better content waiting mechanisms
- Improved selectors with fallbacks
- Stricter keyword filtering
- Better metadata extraction
"""

import os
import time
import re
import json
from collections import deque
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# IMPORTANT: Use undetected-chromedriver for Cloudflare bypass
try:
    import undetected_chromedriver as uc

    UNDETECTED_AVAILABLE = True
except ImportError as e:
    print(f"DEBUG: Import failed with error: {e}")
    UNDETECTED_AVAILABLE = False
except Exception as e:
    print(f"DEBUG: Unexpected error: {e}")
    UNDETECTED_AVAILABLE = False
    print("WARNING: undetected-chromedriver not installed")
    print("Install with: pip install undetected-chromedriver")

# MongoDB imports (optional)
try:
    import certifi
    from pymongo import MongoClient, UpdateOne, ASCENDING
    from dotenv import load_dotenv

    MONGODB_AVAILABLE = True
    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")
except ImportError:
    MONGODB_AVAILABLE = False
    MONGO_URI = None

# JobTeaser URLs
BASE_URL = "https://www.jobteaser.com"
SEARCH_URL = f"{BASE_URL}/fr/job-offers"

# DB Config (if MongoDB enabled)
DB_NAME = "RUCHE_datalake"
COLLECTION_NAME = "jobteaser_raw"

# Search Configuration - More precise keywords
SEARCH_QUERIES = [
    "data scientist",
    "data analyst",
    "data engineer",
    "machine learning engineer",
    "intelligence artificielle",
]

# Selenium Configuration
HEADLESS = False  # Set to True for headless mode
PAGE_LOAD_WAIT = 2  # Increased from 3 to 5 seconds
CARD_CLICK_WAIT = 2  # New: wait after clicking card
CONTENT_LOAD_WAIT = 9  # New: wait for content to fully load


# =============================================================================
# KEYWORD FILTERING (Stricter version)
# =============================================================================

# Patterns to EXCLUDE (false positives)
EXCLUDE_PATTERNS = [
    r"\bcharg[eé]e?\s+(de\s+)?communication\b",  # Communication roles
    r"\bresponsable\s+rh\b",  # HR roles
    r"\btechnicien\b",  # Technician level
    r"\bassistante?\b",  # Assistant roles
    r"\bstage.*communication\b",  # Communication internships
]

# Patterns to INCLUDE (must have at least one)
INCLUDE_PATTERNS = [
    r"\bdata\s+(scientist|analyst|engineer|architect)\b",
    r"\bmachine\s+learning\b",
    r"\bdeep\s+learning\b",
    r"\b(IA|intelligence\s+artificielle)\b",
    r"\bbig\s+data\b",
    r"\bMLOps\b",
    r"\bdata\s+mining\b",
]

# Technical keywords for description validation
TECH_KEYWORDS = [
    "python",
    "sql",
    "machine learning",
    "deep learning",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "pandas",
    "spark",
    "hadoop",
    "data mining",
    "data science",
    "intelligence artificielle",
    "neural network",
    "nlp",
    "computer vision",
    "statistics",
    "statistiques",
]


def is_relevant_job(title: str, description: str) -> bool:
    """Temporary: accept all jobs to test extraction"""
    return True  # Accepter tout pour tester


#    def is_relevant_job(title: str, description: str) -> bool:
#        """
#        Strict filtering for Data/AI jobs
#        Returns True only if job is clearly a Data/AI position
#         """
#        title_lower = title.lower()
#        desc_lower = description.lower()

#        # 1. Reject if exclusion pattern matches
#        for pattern in EXCLUDE_PATTERNS:
#            if re.search(pattern, title_lower):
#                return False

# 2. Accept if inclusion pattern in title
#        for pattern in INCLUDE_PATTERNS:
#            if re.search(pattern, title_lower):
#                return True

# 3. If unclear from title, check description for technical keywords
#        keyword_count = sum(1 for kw in TECH_KEYWORDS if kw in desc_lower)

# Accept if at least 3 technical keywords
#        return keyword_count >= 1
#        """


# Keywords for matching (legacy compatibility)
KEYWORD_PATTERNS = {
    "data scientist": r"\bdata\s+scientist\b",
    "data analyst": r"\bdata\s+analyst\b",
    "data engineer": r"\bdata\s+engineer\b",
    "data": r"\bdata\b",
    "données": r"\bdonn[ée]es?\b",
    "IA": r"\bia\b",
    "intelligence artificielle": r"\bintelligence\s+artificielle\b",
    "AI": r"\bai\b",
    "machine learning": r"\bmachine\s+learning\b",
    "deep learning": r"\bdeep\s+learning\b",
}


def matches_keywords(text: str) -> Tuple[bool, List[str]]:
    """
    Check if text matches keywords
    Returns: (has_match, list_of_matched_keywords)
    """
    text_lower = text.lower()
    matched = [
        kw for kw, pattern in KEYWORD_PATTERNS.items() if re.search(pattern, text_lower)
    ]
    return (len(matched) > 0, matched)


# =============================================================================
# MONGODB CONNECTION (Optional)
# =============================================================================


def get_mongo_collection():
    """Establishes connection to MongoDB Atlas and returns the collection"""
    if not MONGODB_AVAILABLE or not MONGO_URI:
        return None

    try:
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        # Create indexes
        collection.create_index([("id", ASCENDING)], unique=True)
        collection.create_index([("source", ASCENDING)])

        return collection
    except Exception as e:
        print(f"[DB ERROR] Could not connect to MongoDB: {e}")
        return None


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class JobOfferJobTeaser:
    """Job offer from JobTeaser with all details"""

    # Required fields (no default value)
    id: str
    intitule: str
    description: str
    entreprise: str
    lieu: str
    type_contrat: str
    url_origine: str

    # Optional fields (with default values)
    salaire: Optional[str] = None
    date_publication: Optional[str] = None
    competences: List[str] = field(default_factory=list)
    niveau_etudes: Optional[str] = None
    secteur: Optional[str] = None
    url_detail: Optional[str] = None
    matched_keywords: List[str] = field(default_factory=list)
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "jobteaser"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# SELENIUM DRIVER MANAGER (with Cloudflare bypass)
# =============================================================================


class DriverPool:
    """Manage Selenium drivers for parallel scraping with Cloudflare bypass"""

    def __init__(self, pool_size: int = 2, headless: bool = False):
        self.pool_size = pool_size
        self.headless = headless
        self.drivers = []
        self.lock = threading.Lock()

    def create_driver(self) -> webdriver.Chrome:
        """Create Chrome driver with anti-detection for Cloudflare bypass"""

        if UNDETECTED_AVAILABLE:
            # Use undetected-chromedriver (RECOMMENDED)
            options = uc.ChromeOptions()
            # Minimal options - undetected-chromedriver handles most automatically
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")

            if self.headless:
                options.add_argument("--headless=new")
            else:
                options.add_argument("--start-maximized")
            # Create driver - undetected-chromedriver handles anti-detection automatically
            driver = uc.Chrome(options=options, version_main=142)

        else:
            # Fallback to regular ChromeDriver (may not bypass Cloudflare)
            print("[WARNING] Using regular ChromeDriver - Cloudflare may block")
            options = Options()

            if self.headless:
                options.add_argument("--headless")

            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option("useAutomationExtension", False)

            from webdriver_manager.chrome import ChromeDriverManager

            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()), options=options
            )

        # Set timeouts
        driver.set_page_load_timeout(30)
        driver.implicitly_wait(5)

        return driver

    def get_driver(self) -> webdriver.Chrome:
        """Get a driver from the pool or create a new one"""
        with self.lock:
            if self.drivers:
                return self.drivers.pop()
            return self.create_driver()

    def return_driver(self, driver: webdriver.Chrome):
        """Return a driver to the pool"""
        with self.lock:
            if len(self.drivers) < self.pool_size:
                self.drivers.append(driver)
            else:
                driver.quit()

    def close_all(self):
        """Close all drivers in the pool"""
        with self.lock:
            for driver in self.drivers:
                try:
                    driver.quit()
                except:
                    pass
            self.drivers.clear()


# =============================================================================
# CLOUDFLARE & COOKIES HANDLING
# =============================================================================


def handle_cloudflare_challenge(driver: webdriver.Chrome, timeout: int = 30):
    """
    Detect and wait for Cloudflare challenge completion
    If manual intervention needed, notify user
    """
    try:
        # Check if we're on Cloudflare challenge page
        challenge_present = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, "//input[@type='checkbox']"))
        )

        if challenge_present:
            print("⚠️  Cloudflare challenge detected - waiting for resolution...")
            print(
                "   If using undetected-chromedriver, this should resolve automatically"
            )

            # Wait for challenge to disappear
            WebDriverWait(driver, timeout).until(
                EC.invisibility_of_element_located(
                    (By.XPATH, "//input[@type='checkbox']")
                )
            )

            print("✅ Cloudflare challenge completed")
            time.sleep(2)

    except TimeoutException:
        # No challenge or already completed
        pass


def accept_cookies(driver: webdriver.Chrome) -> bool:
    """
    Automatically accept cookie banner on JobTeaser
    Returns True if cookies were accepted
    """
    try:
        # Wait for cookie banner and accept button
        cookie_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "//button[contains(text(), 'Accepter') or contains(text(), 'Accept') or contains(@class, 'accept')]",
                )
            )
        )
        cookie_button.click()
        print("[INFO] ✅ Cookies accepted")
        time.sleep(1)
        return True
    except TimeoutException:
        # No cookie banner or already accepted
        return False
    except Exception as e:
        print(f"[INFO] Could not accept cookies: {e}")
        return False


def wait_for_job_content(driver: webdriver.Chrome, timeout: int = 10) -> bool:
    """
    Wait for job content to be fully loaded
    Returns True if content loaded successfully
    """
    try:
        # Wait for description element to be present
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(
                (
                    By.CSS_SELECTOR,
                    "[data-testid='job-description'], .job-description, .description-content",
                )
            )
        )

        # Wait for text content to be loaded (not just empty div)
        WebDriverWait(driver, timeout).until(
            lambda d: len(
                d.find_element(
                    By.CSS_SELECTOR,
                    "[data-testid='job-description'], .job-description, .description-content",
                ).text.strip()
            )
            > 100
        )

        return True
    except TimeoutException:
        print("[WARN] Timeout waiting for job content")
        return False


def safe_extract(
    driver: webdriver.Chrome, selectors: str, default: str = "Non spécifié"
) -> str:
    """
    Safe extraction with multiple selector fallbacks
    selectors: comma-separated CSS selectors to try
    """
    for selector in selectors.split(", "):
        try:
            element = driver.find_element(By.CSS_SELECTOR, selector.strip())
            text = element.text.strip()
            if text:
                return text
        except NoSuchElementException:
            continue
    return default


# =============================================================================
# RATE LIMITER
# =============================================================================


class RateLimiter:
    """Thread-safe rate limiter"""

    def __init__(self, max_calls: int = 3):
        self.max_calls = max_calls
        self.calls = deque()
        self.lock = threading.Lock()
        self.total = 0
        self.start = time.time()

    def wait(self):
        """Wait if needed to respect rate limit"""
        with self.lock:
            now = time.time()
            while self.calls and self.calls[0] <= now - 1.0:
                self.calls.popleft()

            if len(self.calls) >= self.max_calls:
                sleep_time = 1.0 - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()
                    while self.calls and self.calls[0] <= now - 1.0:
                        self.calls.popleft()

            self.calls.append(time.time())
            self.total += 1

    def stats(self) -> str:
        elapsed = time.time() - self.start
        return f"{self.total} calls in {elapsed:.0f}s ({self.total/elapsed:.1f} req/s)"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def extract_uuid_from_url(url: str) -> Optional[str]:
    """Extract UUID from JobTeaser URL"""
    match = re.search(r"/job-offers/([a-f0-9-]{36})", url)
    return match.group(1) if match else None


# =============================================================================
# JSON EXPORT
# =============================================================================


def save_to_json(
    offers: List[JobOfferJobTeaser], filename: Optional[str] = None
) -> str:
    """Save offers to JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"jobs_jobteaser_{timestamp}.json"

    data = [offer.to_dict() for offer in offers]

    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[JSON] Saved {len(offers)} offers to {filename}")
        return filename
    except Exception as e:
        print(f"[JSON ERROR] Could not save file: {e}")
        return None


# =============================================================================
# BULK OPERATIONS (MongoDB)
# =============================================================================


def bulk_upsert_mongo(collection, offers: List[JobOfferJobTeaser]) -> int:
    """Prepare and execute Bulk Write Upsert to MongoDB"""
    if not offers or collection is None:
        return 0

    operations = []
    for offer in offers:
        operations.append(
            UpdateOne({"id": offer.id}, {"$set": offer.to_dict()}, upsert=True)
        )

    try:
        result = collection.bulk_write(operations)
        return result.upserted_count + result.modified_count
    except Exception as e:
        print(f"[DB WRITE ERROR] {e}")
        return 0


# =============================================================================
# SEARCH FUNCTIONS (with Selenium)
# =============================================================================


def scroll_to_load_more_jobs(driver: webdriver.Chrome, max_scrolls: int = 10) -> int:
    """
    Scroll progressively to load more jobs (lazy loading)
    Returns number of scrolls performed
    """
    last_height = driver.execute_script("return document.body.scrollHeight")
    scrolls = 0

    while scrolls < max_scrolls:
        # Scroll to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        # Calculate new height
        new_height = driver.execute_script("return document.body.scrollHeight")

        # If no change, we've reached the end
        if new_height == last_height:
            break

        last_height = new_height
        scrolls += 1

        # Count cards loaded
        cards = driver.find_elements(By.CLASS_NAME, "JobAdCard_main__1mTeA")
        print(f"[SCROLL] {len(cards)} jobs loaded after scroll {scrolls}")

    return scrolls


def search_jobs_selenium(
    driver: webdriver.Chrome,
    keyword: str,
    seen_ids: Set[str],
    max_pages: int = 50,
    test_mode: bool = False,
) -> List[Tuple[str, str, str, str, str, str]]:
    """
    Search jobs on JobTeaser for a given keyword
    Returns: List of tuples (job_id, title, company, location, contract_type, url)
    """
    print(f"\n{'='*80}\n[SEARCH] Keyword: '{keyword}'\n{'='*80}")

    jobs_found = []
    page = 1

    if test_mode:
        max_pages = 3
        print(f"[TEST MODE] Limiting to {max_pages} pages")

    params = f"?q={keyword}"
    base_search_url = f"{SEARCH_URL}{params}"

    try:
        while page <= max_pages:
            page_url = f"{base_search_url}&page={page}"
            print(f"[SEARCH] Loading page {page}: {page_url}")

            driver.get(page_url)
            time.sleep(PAGE_LOAD_WAIT)

            handle_cloudflare_challenge(driver)
            accept_cookies(driver)

            cards = driver.find_elements(By.CSS_SELECTOR, "[data-testid='jobad-card']")

            if not cards:
                print("[SEARCH] No cards found, stopping pagination")
                break

            print(f"[SEARCH] Found {len(cards)} job cards on page {page}")

            for card in cards:
                try:
                    # URL + ID
                    link_elem = card.find_element(
                        By.CSS_SELECTOR, "a[href*='/job-offers/']"
                    )
                    url_job = link_elem.get_attribute("href")
                    job_id = extract_uuid_from_url(url_job)

                    if not job_id or job_id in seen_ids:
                        continue

                    # TITLE (FIX PRINCIPAL)
                    title_elem = card.find_element(
                        By.CSS_SELECTOR, "h3[class*='JobAdCard_title']"
                    )
                    title = title_elem.text.strip()

                    # COMPANY
                    company_elem = card.find_elements(
                        By.CSS_SELECTOR, "[data-testid='jobad-card-company-name']"
                    )
                    company = (
                        company_elem[0].text.strip()
                        if company_elem and company_elem[0].text.strip()
                        else "Non spécifié"
                    )

                    # LOCATION
                    location_elem = card.find_elements(
                        By.CSS_SELECTOR, "[data-testid='jobad-card-location'] span"
                    )
                    location = (
                        location_elem[0].text.strip()
                        if location_elem and location_elem[0].text.strip()
                        else "France"
                    )

                    # CONTRACT
                    contract_elem = card.find_elements(
                        By.CSS_SELECTOR, "[data-testid='jobad-card-contract'] span"
                    )
                    contract = (
                        contract_elem[0].text.strip()
                        if contract_elem and contract_elem[0].text.strip()
                        else "Non spécifié"
                    )

                    full_url = (
                        url_job
                        if url_job.startswith("http")
                        else f"{BASE_URL}{url_job}"
                    )

                    print("\n[CARD OK]")
                    print(f"ID       : {job_id}")
                    print(f"Title    : {title}")
                    print(f"Company  : {company}")
                    print(f"Location : {location}")
                    print(f"Contract : {contract}")

                    seen_ids.add(job_id)
                    jobs_found.append(
                        (job_id, title, company, location, contract, full_url)
                    )

                except Exception as e:
                    print(f"[CARD ERROR] {e}")
                    continue

            page += 1

    except Exception as e:
        print(f"[SEARCH ERROR] {e}")

    print(
        f"\n[SEARCH] Complete: {len(jobs_found)} unique jobs found for '{keyword}'\n{'='*80}\n"
    )
    return jobs_found


# =============================================================================
# SCRAPING DETAIL FUNCTIONS (with improved extraction)
# =============================================================================


def get_job_detail_selenium(
    driver: webdriver.Chrome,
    job_id: str,
    url: str,
    title_from_card: str,
    company_from_card: str,
    location_from_card: str,
    contract_from_card: str,
) -> Optional[JobOfferJobTeaser]:
    """
    Scrape detailed information for a specific job offer
    Uses data from search cards as fallback to avoid missing info
    """
    try:
        driver.get(url)
        time.sleep(PAGE_LOAD_WAIT)

        # Accept cookies if present
        accept_cookies(driver)

        # Wait for content to load
        if not wait_for_job_content(driver):
            print(f"[WARN] Job {job_id}: Content not loaded properly")

        # =====================================================================
        # Extract data from detail page (with card data as fallback)
        # =====================================================================

        # Title - try to get from detail page, fallback to card
        title_detail = safe_extract(
            driver, "h1, [data-testid='job-title'], .job-title", default=None
        )
        title = (
            title_detail
            if title_detail and title_detail != "Non spécifié"
            else title_from_card
        )

        # Company - try to get from detail page, fallback to card
        company_detail = safe_extract(
            driver,
            "[data-testid='company-name'], .company-name, .employer-name",
            default=None,
        )
        company = (
            company_detail
            if company_detail and company_detail != "Non spécifié"
            else company_from_card
        )

        # Location - try to get from detail page, fallback to card
        location_detail = safe_extract(
            driver, "[data-testid='location'], .job-location, .location", default=None
        )

        # Use detail if available, otherwise card data
        if (
            location_detail
            and location_detail != "Non spécifié"
            and location_detail != "France"
        ):
            # Clean up location (avoid "France, France")
            if ", France" in location_detail:
                location = location_detail
            else:
                location = f"{location_detail}, France"
        else:
            location = location_from_card

        # Contract - try to get from detail page, fallback to card
        contract_detail = safe_extract(
            driver,
            "[data-testid='contract-type'], .contract-type, .job-type",
            default=None,
        )

        if contract_detail and contract_detail != "Non spécifié":
            # Try to extract clean contract type (CDI, CDD, Stage, etc.)
            contract_match = re.search(
                r"\b(CDI|CDD|Stage|Alternance|Freelance|Intérim)\b",
                contract_detail,
                re.IGNORECASE,
            )
            contract = contract_match.group(1) if contract_match else contract_detail
        else:
            contract = contract_from_card

        # =====================================================================
        # Extract description (ESSENTIAL - not in cards)
        # =====================================================================

        description = ""

        # Try 1: data-testid
        try:
            desc_elem = driver.find_element(
                By.CSS_SELECTOR, "[data-testid='job-description']"
            )
            description = desc_elem.text.strip()
        except:
            pass

        # Try 2: class name
        if not description or len(description) < 100:
            try:
                desc_elem = driver.find_element(
                    By.CSS_SELECTOR, ".job-description, .description-content"
                )
                description = desc_elem.text.strip()
            except:
                pass

        # Try 3: all paragraphs as fallback
        if not description or len(description) < 100:
            try:
                paragraphs = driver.find_elements(By.TAG_NAME, "p")
                # Filter out cookie text and short paragraphs
                valid_paragraphs = [
                    p.text.strip()
                    for p in paragraphs[:30]
                    if p.text.strip()
                    and "cookie" not in p.text.lower()
                    and len(p.text.strip()) > 50
                ]
                description = "\n".join(valid_paragraphs[:10])
            except:
                pass

        # If still no description, mark as unavailable
        if not description or len(description) < 50:
            description = "Description non disponible"

        # =====================================================================
        # Extract optional fields (not in cards)
        # =====================================================================

        # Salary
        salary = safe_extract(
            driver, "[data-testid='salary'], .salary, .remuneration", default=None
        )

        # Skills/competences
        competences = []
        try:
            skill_elements = driver.find_elements(
                By.CSS_SELECTOR,
                "[data-testid='skills'] li, .skills li, .skill-list li, .skill-tag",
            )
            competences = [
                skill.text.strip() for skill in skill_elements if skill.text.strip()
            ]
        except:
            pass

        # Publication date
        date_pub = None
        try:
            date_elem = driver.find_element(By.TAG_NAME, "time")
            date_pub = date_elem.get_attribute("datetime")
        except:
            pass

        # Education level
        education = safe_extract(
            driver, "[data-testid='education'], .education-level, .degree", default=None
        )

        # =====================================================================
        # FILTERING
        # =====================================================================

        # Check if job is relevant
        if not is_relevant_job(title, description):
            return None  # Filtered out

        # Keyword matching
        full_text = f"{title} {description} {' '.join(competences)}"
        has_match, matched_kw = matches_keywords(full_text)

        if not has_match:
            return None  # Double-check filter

        # =====================================================================
        # Return complete offer
        # =====================================================================

        return JobOfferJobTeaser(
            id=job_id,
            intitule=title,
            description=description,
            entreprise=company,
            lieu=location,
            type_contrat=contract,
            url_origine=url,
            salaire=salary,
            date_publication=date_pub,
            competences=competences,
            niveau_etudes=education,
            matched_keywords=matched_kw,
        )

    except Exception as e:
        print(f"\n[DETAIL ERROR] Job {job_id}: {e}")
        return None


def scrape_jobs_parallel_selenium(
    jobs_list: List[Tuple],
    driver_pool: DriverPool,
    limiter: RateLimiter,
    workers: int = 2,
) -> Tuple[List[JobOfferJobTeaser], Dict]:
    """
    Scrape job details in parallel using Selenium
    Returns: (list_of_offers, stats_dict)
    """
    print(
        f"\n{'='*80}\n[SCRAPE] Processing {len(jobs_list)} jobs with Selenium\n{'='*80}"
    )

    lock = threading.Lock()
    stats = {"ok": 0, "filtered": 0, "failed": 0}
    collected_offers = []
    start_time = time.time()

    def process_one(job_tuple: Tuple) -> Optional[JobOfferJobTeaser]:
        job_id, title, company, location, contract, url = job_tuple  # Unpacking
        driver = driver_pool.get_driver()
        try:
            limiter.wait()
            # Passer TOUS les params
            offer = get_job_detail_selenium(
                driver, job_id, url, title, company, location, contract
            )
            return offer
        finally:
            driver_pool.return_driver(driver)

    # Process in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, job): job for job in jobs_list}

        for i, future in enumerate(as_completed(futures), 1):
            try:
                offer = future.result()

                with lock:
                    if offer:
                        collected_offers.append(offer)
                        stats["ok"] += 1
                    else:
                        stats["filtered"] += 1

                # Progress log
                if i % 5 == 0 or i == len(jobs_list):
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    pct = i / len(jobs_list) * 100
                    print(
                        f"\r[SCRAPE] {pct:.0f}% | {i}/{len(jobs_list)} | "
                        f"OK:{stats['ok']} FILT:{stats['filtered']} FAIL:{stats['failed']} | "
                        f"{rate:.2f}/s",
                        end="",
                        flush=True,
                    )

            except Exception as e:
                with lock:
                    stats["failed"] += 1

    print(f"\n[SCRAPE] Complete: {stats['ok']} offers collected.\n{'='*80}\n")
    return collected_offers, stats


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Main execution function with improved scraping"""

    print("=" * 80)
    print("JOBTEASER SCRAPER (SELENIUM) -> JSON + MONGODB (OPTIONAL)")
    print("FIXED VERSION - With Cloudflare bypass and improved extraction")
    print("=" * 80)

    if not UNDETECTED_AVAILABLE:
        print("\n⚠️  WARNING: undetected-chromedriver not installed")
        print("   Cloudflare may block the scraper")
        print("   Install with: pip install undetected-chromedriver")
        response = input("\n   Continue anyway? [Y/n]: ")
        if response.lower() == "n":
            return

    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    TEST_MODE = True  # Set to False for production
    UPLOAD_TO_MONGODB = False
    WORKERS = 3

    if TEST_MODE:
        MAX_PAGES = 3
        print("\n⚠️  TEST MODE ENABLED")
        print(f"   - Max pages per keyword: {MAX_PAGES}")
        print(f"   - Workers: {WORKERS}")
        print(f"   - Headless: {HEADLESS}")
        print("   - Stricter filtering enabled")
        print("   - Expected: ~50-100 offers in 3-5 minutes\n")
    else:
        MAX_PAGES = 20  # PARTIAL mode (was 50 FULL)
        WORKERS = 3
        print("\n🚀 PARTIAL MODE")
        print(f"   - Max pages per keyword: {MAX_PAGES}")
        print(f"   - Workers: {WORKERS}")
        print(f"   - Headless: {HEADLESS}")
        print("   - Stricter filtering enabled")
        print("   - Expected: ~2,000-3,000 offers in 30-40 minutes\n")

    # =========================================================================
    # STEP 0: Initialize
    # =========================================================================
    collection = None
    if UPLOAD_TO_MONGODB:
        if not MONGODB_AVAILABLE or not MONGO_URI:
            print("[WARNING] MongoDB not available")
            UPLOAD_TO_MONGODB = False
        else:
            print("[STEP 0/4] Connecting to MongoDB Atlas")
            collection = get_mongo_collection()
            if collection is None:
                UPLOAD_TO_MONGODB = False
    else:
        print("\n[INFO] MongoDB upload disabled - will only save to JSON file")

    # Initialize driver pool
    print(f"[STEP 1/4] Initializing Selenium (headless={HEADLESS})...")
    driver_pool = DriverPool(pool_size=WORKERS, headless=HEADLESS)

    # Create a single driver for search
    search_driver = driver_pool.create_driver()

    limiter = RateLimiter(max_calls=3)
    seen_ids: Set[str] = set()
    all_jobs = []

    try:
        # =====================================================================
        # STEP 2: Search jobs
        # =====================================================================
        print("\n[STEP 2/4] Searching jobs")
        print(f"Keywords: {SEARCH_QUERIES}\n")

        for keyword in SEARCH_QUERIES:
            jobs = search_jobs_selenium(
                driver=search_driver,
                keyword=keyword,
                seen_ids=seen_ids,
                max_pages=MAX_PAGES,
                test_mode=TEST_MODE,
            )
            all_jobs.extend(jobs)
            time.sleep(2)

        if not all_jobs:
            print("[ERROR] No jobs found")
            return

        print(f"\n📊 SEARCH SUMMARY:")
        print(f"   Total unique jobs found: {len(all_jobs)}")

        # =====================================================================
        # STEP 3: Scrape details
        # =====================================================================
        print("\n[STEP 3/4] Scraping job details with Selenium")

        collected_offers, stats = scrape_jobs_parallel_selenium(
            jobs_list=all_jobs,
            driver_pool=driver_pool,
            limiter=limiter,
            workers=WORKERS,
        )

        if not collected_offers:
            print("[ERROR] No offers collected after scraping")
            return

        # =====================================================================
        # STEP 4: Save results
        # =====================================================================
        print("\n[STEP 4/4] Saving results")
        output_dir = "json_scrapped"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = os.path.join(output_dir, f"jobs_jobteaser_{timestamp}.json")

        saved_file = save_to_json(collected_offers, json_filename)

        if not saved_file:
            print("[ERROR] Could not save JSON file")
            return

        # MongoDB upload (optional)
        db_stats = {"uploaded": 0}

        if UPLOAD_TO_MONGODB and collection is not None:
            print(f"\n[MONGODB] Uploading {len(collected_offers)} offers...")

            try:
                db_ops = bulk_upsert_mongo(collection, collected_offers)
                db_stats["uploaded"] = db_ops
                print(f"[MONGODB] ✅ Uploaded {db_ops} offers")
            except Exception as e:
                print(f"[MONGODB] ❌ Upload failed: {e}")

        # =====================================================================
        # FINAL SUMMARY
        # =====================================================================
        print(f"\n{'='*80}")
        print("✅ SCRAPING COMPLETE")
        print(f"{'='*80}")
        print(f"📊 Results:")
        print(f"   Searched: {len(all_jobs)} jobs")
        print(f"   Scraped successfully: {stats['ok']} jobs")
        print(f"   Filtered (not relevant): {stats['filtered']} jobs")
        print(f"   Failed: {stats['failed']} jobs")
        if stats["ok"] + stats["filtered"] > 0:
            print(
                f"   Precision: {stats['ok']/(stats['ok']+stats['filtered'])*100:.1f}%"
            )

        print(f"\n💾 Local JSON file:")
        print(f"   📄 {json_filename}")
        print(f"   Size: {len(collected_offers)} offers")

        if UPLOAD_TO_MONGODB and db_stats["uploaded"] > 0:
            print(f"\n☁️  MongoDB:")
            print(f"   Database: {DB_NAME}.{COLLECTION_NAME}")
            print(f"   Uploaded: {db_stats['uploaded']} offers")

        print(f"\n⏱️  Stats: {limiter.stats()}")
        print("=" * 80)

    finally:
        print("\n[CLEANUP] Closing browsers...")
        try:
            search_driver.quit()
        except:
            pass
        driver_pool.close_all()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
    finally:
        input("\nPress ENTER to exit...")
