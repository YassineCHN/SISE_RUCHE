"""
JobTeaser Scraper -> MongoDB Atlas
Scrapes job offers containing "data" OR "intelligence artificielle" with deduplication and Upsert strategy
Aligned with France Travail methodology (broad search + strict filtering)
"""

import os
import time
import re
import threading
import certifi
from collections import deque
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne, ASCENDING

# Load environment
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")  # URL avec votre username et mpd dans votre .env

# JobTeaser URLs
BASE_URL = "https://www.jobteaser.com"
SEARCH_URL = f"{BASE_URL}/fr/job-offers"

# DB Config
DB_NAME = "RUCHE_datalake"
COLLECTION_NAME = "jobteaser_raw"

# Search Configuration
SEARCH_QUERIES = [
    "data",
    "intelligence artificielle"
]

# =============================================================================
# MONGODB CONNECTION
# =============================================================================

def get_mongo_collection():
    """Establishes connection to MongoDB Atlas and returns the collection"""
    try:
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Création d'index uniques
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
    id: str                          # UUID extracted from URL
    intitule: str
    description: str
    entreprise: str
    lieu: str                        # "Paris, France"
    type_contrat: str                # "Stage 4 à 6 mois", "CDI", etc.
    
    # Optional fields
    salaire: Optional[str] = None
    date_publication: Optional[str] = None
    competences: List[str] = field(default_factory=list)
    niveau_etudes: Optional[str] = None
    secteur: Optional[str] = None
    url_detail: Optional[str] = None
    
    # Metadata
    url_origine: str
    matched_keywords: List[str] = field(default_factory=list)
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "jobteaser"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """Thread-safe rate limiter"""
    
    def __init__(self, max_calls: int = 5):
        self.max_calls = max_calls  # Conservative: 5 req/s
        self.calls = deque()
        self.lock = threading.Lock()
        self.total = 0
        self.start = time.time()
    
    def wait(self):
        """Wait if needed to respect rate limit"""
        with self.lock:
            now = time.time()
            # Remove calls older than 1 second
            while self.calls and self.calls[0] <= now - 1.0:
                self.calls.popleft()
            
            # If at limit, wait
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
# KEYWORD MATCHING (Aligned with France Travail)
# =============================================================================

KEYWORD_PATTERNS = {
    'data': r'\bdata\b',
    'données': r'\bdonn[ée]es?\b',
    'IA': r'\bia\b',
    'intelligence artificielle': r'\bintelligence\s+artificielle\b',
    'AI': r'\bai\b',
    'machine learning': r'\bmachine\s+learning\b',
    'deep learning': r'\bdeep\s+learning\b'
}

def matches_keywords(text: str) -> Tuple[bool, List[str]]:
    """
    Check if text matches keywords (same as France Travail)
    Returns: (has_match, list_of_matched_keywords)
    """
    text_lower = text.lower()
    matched = [kw for kw, pattern in KEYWORD_PATTERNS.items() 
               if re.search(pattern, text_lower)]
    return (len(matched) > 0, matched)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_uuid_from_url(url: str) -> Optional[str]:
    """
    Extract UUID from JobTeaser URL
    Example: /fr/job-offers/bf55098a-e993-4f03-ab63-25542dd1b001-vinci-...
    Returns: bf55098a-e993-4f03-ab63-25542dd1b001
    """
    match = re.search(r'/job-offers/([a-f0-9-]{36})', url)
    return match.group(1) if match else None


def get_headers() -> Dict[str, str]:
    """Return headers to simulate a real browser"""
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0"
    }


# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================

def search_jobs(keyword: str, limiter: RateLimiter, seen_ids: Set[str], 
                max_pages: int = 50, test_mode: bool = False) -> List[Tuple[str, str, str, str, str]]:
    """
    Search jobs on JobTeaser for a given keyword with pagination
    
    Returns: List of tuples (job_id, title, company, location, contract_type, url)
    """
    print(f"\n{'='*80}\n[SEARCH] Keyword: '{keyword}'\n{'='*80}")
    
    jobs_found = []
    page = 1
    
    # In test mode, limit to fewer pages
    if test_mode:
        max_pages = 3
        print(f"[TEST MODE] Limiting to {max_pages} pages")
    
    while page <= max_pages:
        limiter.wait()
        
        params = {
            "q": keyword,
            "page": page,
            "lat": "46.711046499999995",
            "lng": "2.1811786692949857",
            "localized_location": "France",
            "location": "France::_Y291bnRyeTo6OnVGaW9mQWV3VEVWbzlSc056bVZmZU5jOEFyTT0="
        }
        
        try:
            response = requests.get(SEARCH_URL, params=params, headers=get_headers(), timeout=15)
            
            if response.status_code != 200:
                print(f"[SEARCH] Page {page}: Status {response.status_code}")
                break
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Find all job cards
            cards = soup.find_all("div", {"class": "JobAdCard_main__1mTeA"})
            
            if not cards:
                print(f"[SEARCH] Page {page}: No more results")
                break
            
            page_new = 0
            for card in cards:
                try:
                    # Extract link
                    link_elem = card.find("a", {"class": "JobAdCard_link__LMtBN"})
                    if not link_elem or 'href' not in link_elem.attrs:
                        continue
                    
                    url = link_elem['href']
                    job_id = extract_uuid_from_url(url)
                    
                    if not job_id or job_id in seen_ids:
                        continue
                    
                    # Extract data
                    title = link_elem.get_text(strip=True)
                    
                    company_elem = card.find("p", {"data-testid": "jobad-card-company-name"})
                    company = company_elem.get_text(strip=True) if company_elem else "Non spécifié"
                    
                    location_elem = card.find("span", {"class": "sk-Typography_body1__rvFYv"}, 
                                             string=lambda t: t and "France" in t)
                    location = location_elem.get_text(strip=True) if location_elem else "France"
                    
                    contract_elem = card.find("div", {"data-testid": "jobad-card-contract"})
                    if contract_elem:
                        contract_span = contract_elem.find("span")
                        contract = contract_span.get_text(strip=True) if contract_span else "Non spécifié"
                    else:
                        contract = "Non spécifié"
                    
                    full_url = f"{BASE_URL}{url}" if url.startswith("/") else url
                    
                    seen_ids.add(job_id)
                    jobs_found.append((job_id, title, company, location, contract, full_url))
                    page_new += 1
                    
                except Exception as e:
                    print(f"[SEARCH] Error parsing card: {e}")
                    continue
            
            print(f"[SEARCH] Page {page}: {len(cards)} cards, {page_new} new jobs")
            
            # Check if there's a next page
            pagination = soup.find("nav", {"data-testid": "job-ads-pagination"})
            if not pagination:
                print(f"[SEARCH] No pagination found, stopping")
                break
            
            # Look for next page button (not disabled)
            next_button = pagination.find("a", {"aria-label": "Aller à la page suivante"})
            if not next_button:
                print(f"[SEARCH] No next page button, stopping")
                break
            
            page += 1
            time.sleep(0.5)  # Politeness delay
            
        except Exception as e:
            print(f"[SEARCH] Error on page {page}: {e}")
            break
    
    print(f"[SEARCH] Complete: {len(jobs_found)} unique jobs found for '{keyword}'\n{'='*80}\n")
    return jobs_found


# =============================================================================
# SCRAPING DETAIL FUNCTIONS
# =============================================================================

def get_job_detail(job_id: str, url: str, limiter: RateLimiter) -> Optional[JobOfferJobTeaser]:
    """
    Scrape detailed information for a specific job offer
    """
    limiter.wait()
    
    try:
        response = requests.get(url, headers=get_headers(), timeout=15)
        
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Extract title
        title_elem = soup.find("h1")
        title = title_elem.get_text(strip=True) if title_elem else "Titre non disponible"
        
        # Extract company
        company_elem = soup.find("div", {"data-testid": "company-name"}) or \
                      soup.find("a", href=lambda h: h and "/companies/" in h)
        company = company_elem.get_text(strip=True) if company_elem else "Entreprise non spécifiée"
        
        # Extract location
        location_elem = soup.find("div", {"data-testid": "location"}) or \
                       soup.find("span", string=lambda t: t and "France" in t)
        location = location_elem.get_text(strip=True) if location_elem else "France"
        
        # Extract contract type
        contract_elem = soup.find("div", {"data-testid": "contract-type"})
        contract = contract_elem.get_text(strip=True) if contract_elem else "Non spécifié"
        
        # Extract description
        description_elem = soup.find("div", {"data-testid": "job-description"}) or \
                          soup.find("div", class_=lambda c: c and "description" in c.lower())
        
        if description_elem:
            # Get all text, clean it
            description = description_elem.get_text(separator="\n", strip=True)
        else:
            # Fallback: get all paragraph text
            paragraphs = soup.find_all("p")
            description = "\n".join([p.get_text(strip=True) for p in paragraphs[:20]])
        
        if not description or len(description) < 50:
            description = "Description non disponible"
        
        # Extract salary (if available)
        salary_elem = soup.find("div", {"data-testid": "salary"})
        salary = salary_elem.get_text(strip=True) if salary_elem else None
        
        # Extract competences/skills
        competences = []
        skills_section = soup.find("div", {"data-testid": "skills"}) or \
                        soup.find("ul", class_=lambda c: c and "skill" in c.lower())
        if skills_section:
            skill_items = skills_section.find_all("li")
            competences = [skill.get_text(strip=True) for skill in skill_items]
        
        # Extract publication date
        date_elem = soup.find("time")
        date_pub = date_elem.get("datetime") if date_elem and date_elem.has_attr("datetime") else None
        
        # Check keywords
        full_text = f"{title} {description} {' '.join(competences)}"
        has_match, matched_kw = matches_keywords(full_text)
        
        if not has_match:
            return None  # Filtered out
        
        return JobOfferJobTeaser(
            id=job_id,
            intitule=title,
            description=description,
            entreprise=company,
            lieu=location,
            type_contrat=contract,
            salaire=salary,
            date_publication=date_pub,
            competences=competences,
            url_origine=url,
            matched_keywords=matched_kw
        )
        
    except Exception as e:
        print(f"[DETAIL ERROR] Job {job_id}: {e}")
        return None


# =============================================================================
# BULK OPERATIONS
# =============================================================================

def bulk_upsert_mongo(collection, offers: List[JobOfferJobTeaser]) -> int:
    """Prepare and execute Bulk Write Upsert to MongoDB"""
    if not offers or collection is None:
        return 0

    operations = []
    for offer in offers:
        operations.append(
            UpdateOne(
                {"id": offer.id},
                {"$set": offer.to_dict()},
                upsert=True
            )
        )
    
    try:
        result = collection.bulk_write(operations)
        return result.upserted_count + result.modified_count
    except Exception as e:
        print(f"[DB WRITE ERROR] {e}")
        return 0


def scrape_to_mongo(jobs_list: List[Tuple], limiter: RateLimiter, 
                   collection, workers: int = 8, verify: bool = True):
    """
    Scrape job details in parallel and push directly to MongoDB
    """
    print(f"\n{'='*80}\n[SCRAPE] Processing {len(jobs_list)} jobs -> MongoDB\n{'='*80}")
    
    lock = threading.Lock()
    stats = {'ok': 0, 'filtered': 0, 'failed': 0, 'db_ops': 0}
    buffer = []
    start_time = time.time()
    
    def process_one(job_tuple: Tuple) -> Optional[JobOfferJobTeaser]:
        job_id, title, company, location, contract, url = job_tuple
        offer = get_job_detail(job_id, url, limiter)
        return offer
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, job): job for job in jobs_list}
        
        for i, future in enumerate(as_completed(futures), 1):
            try:
                offer = future.result()
                
                with lock:
                    if offer:
                        buffer.append(offer)
                        stats['ok'] += 1
                        
                        # Bulk insert every 20 offers
                        if len(buffer) >= 20:
                            ops_count = bulk_upsert_mongo(collection, buffer)
                            stats['db_ops'] += ops_count
                            buffer.clear()
                    else:
                        stats['filtered' if verify else 'failed'] += 1
                
                # Progress log
                if i % 10 == 0 or i == len(jobs_list):
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    pct = i / len(jobs_list) * 100
                    print(f"\r[SCRAPE] {pct:.0f}% | {i}/{len(jobs_list)} | "
                          f"OK:{stats['ok']} FILT:{stats['filtered']} | "
                          f"DB Ops:{stats['db_ops']} | {rate:.1f}/s", end="", flush=True)
                          
            except Exception as e:
                stats['failed'] += 1
                print(f"\n[ERROR] {e}")
    
    # Flush remaining buffer
    if buffer:
        print("\n[DB] Flushing remaining buffer...")
        ops_count = bulk_upsert_mongo(collection, buffer)
        stats['db_ops'] += ops_count

    print(f"\n[SCRAPE] Complete: {stats['ok']} offers processed -> MongoDB.\n{'='*80}\n")
    return stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Main execution function
    
    Modes:
    - TEST_MODE = True: Quick test with limited results (3 pages, 2 workers)
    - TEST_MODE = False: Full production scraping (all pages, 8 workers)
    """
    
    print("="*80)
    print("JOBTEASER SCRAPER -> MONGODB ATLAS")
    print("Strategy: Broad search + Strict filtering (aligned with France Travail)")
    print("="*80)
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    TEST_MODE = True  # Set to False for production
    
    if TEST_MODE:
        MAX_PAGES = 3
        WORKERS = 2
        print("\n⚠️  TEST MODE ENABLED")
        print(f"   - Max pages per keyword: {MAX_PAGES}")
        print(f"   - Workers: {WORKERS}")
        print("   - Expected: ~50-100 offers\n")
    else:
        MAX_PAGES = 50
        WORKERS = 8
        print("\n🚀 PRODUCTION MODE")
        print(f"   - Max pages per keyword: {MAX_PAGES}")
        print(f"   - Workers: {WORKERS}")
        print("   - Expected: ~12,000-15,000 offers\n")
    
    # =========================================================================
    # STEP 0: Connect to MongoDB
    # =========================================================================
    if not MONGO_URI:
        print("[ERROR] Missing MONGO_URI in environment variables")
        return
    
    print("[STEP 0/3] Connecting to MongoDB Atlas")
    collection = get_mongo_collection()
    if collection is None:
        return
    
    limiter = RateLimiter(max_calls=5)  # Conservative rate limit
    seen_ids: Set[str] = set()
    all_jobs = []
    
    # =========================================================================
    # STEP 1: Search jobs for all keywords
    # =========================================================================
    print("\n[STEP 1/3] Searching jobs")
    print(f"Keywords: {SEARCH_QUERIES}\n")
    
    for keyword in SEARCH_QUERIES:
        jobs = search_jobs(
            keyword=keyword,
            limiter=limiter,
            seen_ids=seen_ids,
            max_pages=MAX_PAGES,
            test_mode=TEST_MODE
        )
        all_jobs.extend(jobs)
        time.sleep(1)  # Politeness delay between keywords
    
    if not all_jobs:
        print("[ERROR] No jobs found")
        return
    
    print(f"\n📊 SEARCH SUMMARY:")
    print(f"   Total unique jobs found: {len(all_jobs)}")
    print(f"   Duplicate IDs filtered: {limiter.total - len(all_jobs)}")
    
    # =========================================================================
    # STEP 2: Confirmation (if large volume)
    # =========================================================================
    if len(all_jobs) > 100 and not TEST_MODE:
        print(f"\n⚠️  WARNING: About to scrape {len(all_jobs)} job details")
        print(f"   Estimated time: {len(all_jobs) / WORKERS / 5 / 60:.1f} minutes")
        response = input("   Continue? [Y/n]: ")
        if response.lower() == 'n':
            print("Aborted by user")
            return
    
    # =========================================================================
    # STEP 3: Scrape details & Upload to MongoDB
    # =========================================================================
    print("\n[STEP 3/3] Scraping job details & Upserting to MongoDB")
    
    stats = scrape_to_mongo(
        jobs_list=all_jobs,
        limiter=limiter,
        collection=collection,
        workers=WORKERS,
        verify=True
    )
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print(f"\n{'='*80}")
    print("✅ SCRAPING COMPLETE")
    print(f"{'='*80}")
    print(f"📊 Results:")
    print(f"   Searched: {len(all_jobs)} jobs")
    print(f"   Scraped successfully: {stats['ok']} jobs")
    print(f"   Filtered (no keywords): {stats['filtered']} jobs")
    print(f"   Failed: {stats['failed']} jobs")
    print(f"   Precision: {stats['ok']/(stats['ok']+stats['filtered'])*100:.1f}%")
    print(f"\n💾 Database: {DB_NAME}.{COLLECTION_NAME}")
    print(f"   Total operations: {stats['db_ops']}")
    print(f"\n⏱️  API Stats: {limiter.stats()}")
    print("="*80)


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
