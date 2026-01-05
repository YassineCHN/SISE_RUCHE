"""
Service Public Job Scraper - OPTIMIZED & WORKING VERSION
"""

import os
import re
import time
import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any, Set
from urllib.parse import urljoin
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Import MongoDB utilities
from mongodb.mongodb_utils import (
    get_collection,
    create_unique_index,
    bulk_upsert,
    count_documents,
    get_collection_stats,
)

# Load environment
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Collection name
COLLECTION_NAME = "servicepublic_raw"

# Base URL
BASE_URL = "https://choisirleservicepublic.gouv.fr"


# =============================================================================
# RATE LIMITER
# =============================================================================


class RateLimiter:
    """Thread-safe rate limiter for API calls"""

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
        """Get rate limiter statistics"""
        elapsed = time.time() - self.start
        return f"{self.total} calls in {elapsed:.0f}s ({self.total/elapsed:.1f} req/s)"


# =============================================================================
# DATA MODEL
# =============================================================================


@dataclass
class ServicePublicJobOffer:
    """Service Public job offer - harmonized schema"""

    id: str
    intitule: str
    description: str
    lieu: str
    type_contrat: str
    date_creation: str

    entreprise: Optional[str] = None
    salaire: Optional[str] = None
    experience: Optional[str] = None
    competences: List[str] = field(default_factory=list)
    formations: List[str] = field(default_factory=list)
    langues: List[str] = field(default_factory=list)
    permis: List[str] = field(default_factory=list)
    url_origine: Optional[str] = None
    matched_keywords: List[str] = field(default_factory=list)

    grade: Optional[str] = None
    departement: Optional[str] = None
    date_limite: Optional[str] = None
    avantages: Optional[str] = None
    teletravail: Optional[str] = None

    source: str = "choisirleservicepublic.gouv.fr"
    mot_cle: Optional[str] = None
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# JSON EXTRACTION
# =============================================================================


def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Extract JSON object from text with multiple strategies
    """
    try:
        # Strategy 1: Direct parsing
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code blocks
        markdown_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        markdown_match = re.search(markdown_pattern, text, re.DOTALL)

        if markdown_match:
            json_text = markdown_match.group(1).strip()
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find JSON object by braces
        first_brace = text.find("{")
        last_brace = text.rfind("}")

        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_text = text[first_brace : last_brace + 1]
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                pass

        return None

    except Exception:
        return None


# =============================================================================
# HTML EXTRACTION
# =============================================================================


def extract_relevant_html(html_content: str) -> str:
    """Extract only relevant content from HTML"""
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove unnecessary tags
    for tag in soup(
        ["script", "style", "nav", "header", "footer", "aside", "form", "button"]
    ):
        tag.decompose()

    # Try to find main content area
    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_=re.compile(r"content|main|body|offer|offre", re.I))
        or soup.find("div", id=re.compile(r"content|main|body|offer|offre", re.I))
    )

    if main_content:
        text = main_content.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    # Clean text
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = "\n".join(lines)

    # Intelligent limit
    if len(text) > 8000:
        text = text[:4000] + "\n...\n" + text[-4000:]

    return text


# =============================================================================
# LLM EXTRACTION
# =============================================================================


def extract_with_llm(
    html_content: str, url: str, keyword: str, limiter: RateLimiter
) -> tuple[Optional[ServicePublicJobOffer], Optional[str]]:
    """
    LLM extraction with IMPROVED PROMPT (few-shot learning)
    """
    try:
        # Extract relevant content
        text_content = extract_relevant_html(html_content)

        if not text_content or len(text_content) < 100:
            return (None, "empty_content")

        # Rate limiting
        limiter.wait()

        # ✅ IMPROVED PROMPT with example
        prompt = f"""Tu es un parseur JSON. Extrait UNIQUEMENT le JSON, rien d'autre.

EXEMPLE D'ENTRÉE :
"Référence: ABC123. Poste de Data Scientist à Paris pour le Ministère. Salaire 45K. Compétences: Python, SQL."

EXEMPLE DE SORTIE VALIDE :
{{"id":"ABC123","titre":"Data Scientist","collectivite":"Ministère","departement":"75","lieu":"Paris","grade":"","type_emploi":"","salaire":"45000","date_publication":"","date_limite":"","competences":"Python, SQL","experience":"","avantages":"","teletravail":"","description":"Poste de Data Scientist"}}

RÈGLES ABSOLUES :
1. Réponds UNIQUEMENT avec le JSON (pas de "Voici", pas de markdown, pas d'explication)
2. Si une info manque, mets ""
3. Compétences en string séparées par virgules (pas de liste)

TEXTE À PARSER :
{text_content[:6000]}

TON JSON (rien d'autre) :"""

        # API call
        try:
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {MISTRAL_API_KEY}",
                },
                json={
                    "model": "mistral-large-latest",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,  # ← Réduit à 0 (déterministe)
                    "max_tokens": 600,  # ← Réduit (force concision)
                },
                timeout=30,
            )
        except requests.Timeout:
            return (None, "timeout")
        except requests.RequestException:
            return (None, "llm_api_error")

        # Check status
        if response.status_code == 429:
            time.sleep(3)
            return (None, "rate_limit_429")
        elif response.status_code != 200:
            return (None, "llm_api_error")

        # Parse API response
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            return (None, "llm_api_error")

        if (
            "choices" not in response_data
            or len(response_data["choices"]) == 0
            or "message" not in response_data["choices"][0]
            or "content" not in response_data["choices"][0]["message"]
        ):
            return (None, "llm_api_error")

        text_response = response_data["choices"][0]["message"]["content"]

        if not text_response or text_response.strip() == "":
            return (None, "empty_content")

        # ✅ AGGRESSIVE JSON EXTRACTION
        llm_data = extract_json_from_text(text_response)

        if llm_data is None:
            # ✅ FALLBACK : Try even more aggressive extraction
            llm_data = extract_json_aggressive(text_response)

        if llm_data is None:
            return (None, "json_parse_error")

        if not isinstance(llm_data, dict) or len(llm_data) == 0:
            return (None, "json_parse_error")

        # Transform to harmonized schema
        offer = transform_to_harmonized_schema(llm_data, url, keyword)
        return (offer, None)  # Success

    except Exception as e:
        return (None, "other")


def transform_to_harmonized_schema(
    llm_data: dict, url: str, keyword: str
) -> ServicePublicJobOffer:
    """
    Transform LLM output to harmonized schema
    Handles both string and list formats for competences
    """

    # ✅ PARSE COMPETENCES (gérer string ET liste)
    competences_raw = llm_data.get("competences", "")

    if isinstance(competences_raw, list):
        # Déjà une liste : garder tel quel
        competences = [str(c).strip() for c in competences_raw if c]
    elif isinstance(competences_raw, str):
        # String : split par virgule
        competences = [c.strip() for c in competences_raw.split(",") if c.strip()]
    else:
        # Autre type : liste vide
        competences = []

    # Convert dates
    date_pub = llm_data.get("date_publication", "")
    date_iso = convert_date_to_iso(date_pub) if isinstance(date_pub, str) else ""

    date_limite = llm_data.get("date_limite", "")
    date_limite_iso = (
        convert_date_to_iso(date_limite) if isinstance(date_limite, str) else ""
    )

    # Detect keywords
    titre = str(llm_data.get("titre", ""))
    description = str(llm_data.get("description", ""))
    competences_text = " ".join(competences)
    full_text = f"{titre} {description} {competences_text}".lower()

    matched_keywords = []
    if "data" in full_text or "données" in full_text:
        matched_keywords.append("data")
    if "ia" in full_text or "intelligence artificielle" in full_text:
        matched_keywords.append("ia")

    return ServicePublicJobOffer(
        id=str(llm_data.get("id", "")),
        intitule=titre,
        description=description,
        lieu=str(llm_data.get("lieu", "")),
        type_contrat=str(llm_data.get("type_emploi", "")),
        date_creation=date_iso,
        entreprise=(
            llm_data.get("collectivite") if llm_data.get("collectivite") else None
        ),
        salaire=llm_data.get("salaire") if llm_data.get("salaire") else None,
        experience=llm_data.get("experience") if llm_data.get("experience") else None,
        competences=competences,
        formations=[],
        langues=[],
        permis=[],
        url_origine=url,
        matched_keywords=matched_keywords,
        grade=llm_data.get("grade") if llm_data.get("grade") else None,
        departement=(
            llm_data.get("departement") if llm_data.get("departement") else None
        ),
        date_limite=date_limite_iso,
        avantages=llm_data.get("avantages") if llm_data.get("avantages") else None,
        teletravail=(
            llm_data.get("teletravail") if llm_data.get("teletravail") else None
        ),
        mot_cle=keyword,
    )


def convert_date_to_iso(date_str: str) -> str:
    """Convert French date to ISO format"""
    if not date_str:
        return ""

    try:
        # DD/MM/YYYY
        if "/" in date_str:
            parts = date_str.split("/")
            if len(parts) == 3:
                day, month, year = parts
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        # French month names
        months_fr = {
            "janvier": "01",
            "février": "02",
            "mars": "03",
            "avril": "04",
            "mai": "05",
            "juin": "06",
            "juillet": "07",
            "août": "08",
            "septembre": "09",
            "octobre": "10",
            "novembre": "11",
            "décembre": "12",
        }

        for month_name, month_num in months_fr.items():
            if month_name in date_str.lower():
                match = re.search(
                    r"(\d{1,2})\s+" + month_name + r"\s+(\d{4})", date_str.lower()
                )
                if match:
                    day = match.group(1).zfill(2)
                    year = match.group(2)
                    return f"{year}-{month_num}-{day}"

        return ""

    except:
        return ""


# =============================================================================
# WEB SCRAPING
# =============================================================================


def get_total_pages(session: requests.Session, search_url: str) -> int:
    """Get total number of result pages"""
    try:
        response = session.get(search_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        page_numbers = []
        pagination_links = soup.find_all("a", class_="fr-pagination__link")

        for link in pagination_links:
            text = link.get_text(strip=True)
            if text.isdigit():
                page_numbers.append(int(text))

        return max(page_numbers) if page_numbers else 1
    except Exception as e:
        print(f"[ERROR] Cannot detect page count: {e}")
        return 1


def extract_job_links(page_soup: BeautifulSoup) -> List[str]:
    """Extract job offer URLs from search results page"""
    job_urls = []
    offre_blocks = page_soup.find_all("div", class_="fr-card") or page_soup.find_all(
        "article"
    )

    for block in offre_blocks:
        link_tag = block.find("a", href=True)
        if link_tag and "/offre-emploi/" in link_tag["href"]:
            full_url = urljoin(BASE_URL, link_tag["href"])
            job_urls.append(full_url)

    return job_urls


# =============================================================================
# PARALLEL SCRAPER
# =============================================================================


def scrape_keyword_parallel(
    keyword: str, collection, workers: int = 4, max_jobs: Optional[int] = None
):
    """
    Scrape with parallel processing and detailed error tracking
    """
    print(f"\n{'='*80}")
    print(f"[SCRAPE PARALLEL] Keyword: '{keyword.upper()}' | Workers: {workers}")
    print(f"{'='*80}")

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )

    # Rate limiter
    limiter = RateLimiter(max_calls=3)

    start_url = f"{BASE_URL}/nos-offres/filtres/mot-cles/{keyword}/"
    total_pages = get_total_pages(session, start_url)

    print(f"[INFO] Pages to scrape: {total_pages}")

    # Collect URLs
    all_job_urls = []

    for page_num in range(1, total_pages + 1):
        if max_jobs and len(all_job_urls) >= max_jobs:
            break

        current_url = start_url if page_num == 1 else f"{start_url}page/{page_num}/"
        print(f"[PAGE {page_num}/{total_pages}] Collecting URLs...")

        try:
            response = session.get(current_url)
            soup = BeautifulSoup(response.content, "html.parser")
            job_urls = extract_job_links(soup)
            all_job_urls.extend(job_urls)

            print(f"  Found {len(job_urls)} URLs (Total: {len(all_job_urls)})")

        except Exception as e:
            print(f"[ERROR] Page {page_num}: {e}")

    if max_jobs:
        all_job_urls = all_job_urls[:max_jobs]

    print(
        f"\n[INFO] Collected {len(all_job_urls)} job URLs. Starting parallel extraction..."
    )

    # Parallel extraction
    lock = threading.Lock()
    buffer = []
    seen_hashes = set()
    stats = {"success": 0, "failed": 0, "duplicates": 0, "db_ops": 0}
    error_details = {
        "rate_limit_429": 0,
        "timeout": 0,
        "http_error": 0,
        "empty_content": 0,
        "json_parse_error": 0,
        "llm_api_error": 0,
        "other": 0,
    }

    start_time = time.time()

    def process_one_job(job_url: str) -> Optional[ServicePublicJobOffer]:
        """Process single job with detailed error tracking"""
        worker_session = requests.Session()
        worker_session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )

        try:
            # Fetch HTML
            job_response = worker_session.get(job_url, timeout=15)

            if job_response.status_code != 200:
                with lock:
                    error_details["http_error"] += 1
                return None

            # Extract with LLM
            offer, error_type = extract_with_llm(
                job_response.content, job_url, keyword, limiter
            )

            # Track error type
            if offer is None and error_type:
                with lock:
                    error_details[error_type] = error_details.get(error_type, 0) + 1

            return offer

        except requests.Timeout:
            with lock:
                error_details["timeout"] += 1
            return None
        except Exception as e:
            with lock:
                error_details["other"] += 1
            print(
                f"    [EXCEPTION in process_one_job] {type(e).__name__}: {str(e)[:100]}"
            )
            return None
        finally:
            worker_session.close()

    # Launch parallel processing
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one_job, url): url for url in all_job_urls}

        for i, future in enumerate(as_completed(futures), 1):
            try:
                offer = future.result()

                with lock:
                    if offer:
                        job_hash = hashlib.md5(
                            f"{offer.intitule}_{offer.entreprise}_{offer.lieu}".encode()
                        ).hexdigest()

                        if job_hash not in seen_hashes:
                            seen_hashes.add(job_hash)
                            buffer.append(offer.to_dict())
                            stats["success"] += 1

                            if len(buffer) >= 20:
                                ops = bulk_upsert(collection, buffer)
                                stats["db_ops"] += ops
                                buffer.clear()
                        else:
                            stats["duplicates"] += 1
                    else:
                        stats["failed"] += 1

                    # Progress
                    if i % 10 == 0 or i == len(all_job_urls):
                        elapsed = time.time() - start_time
                        rate = i / elapsed if elapsed > 0 else 0
                        pct = i / len(all_job_urls) * 100

                        print(
                            f"\r[PROGRESS] {pct:.0f}% | {i}/{len(all_job_urls)} | "
                            f"✅ {stats['success']} | ❌ {stats['failed']} | "
                            f"⚠️ {stats['duplicates']} | 💾 {stats['db_ops']} | "
                            f"{rate:.1f}/s",
                            end="",
                            flush=True,
                        )

            except Exception as e:
                stats["failed"] += 1
                print(f"\n[EXCEPTION in main loop] {type(e).__name__}: {str(e)[:100]}")

    # Final flush
    if buffer:
        ops = bulk_upsert(collection, buffer)
        stats["db_ops"] += ops
        print(f"\n[DB] Final flush: {ops} documents")

    # Statistics
    elapsed_total = time.time() - start_time

    print(f"\n\n{'='*80}")
    print(f"[COMPLETE] Keyword '{keyword}'")
    print(f"{'='*80}")
    print(f"Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"Processed: {len(all_job_urls)} URLs")
    print(
        f"Success: {stats['success']} ({stats['success']/len(all_job_urls)*100:.1f}%)"
    )
    print(f"Failed: {stats['failed']} ({stats['failed']/len(all_job_urls)*100:.1f}%)")
    print(f"Duplicates: {stats['duplicates']}")
    print(f"DB operations: {stats['db_ops']}")
    print(f"API calls: {limiter.stats()}")

    # Detailed error breakdown
    print(f"\n[ERROR BREAKDOWN]")
    for error_type, count in sorted(
        error_details.items(), key=lambda x: x[1], reverse=True
    ):
        if count > 0:
            print(f"  {error_type}: {count} ({count/len(all_job_urls)*100:.1f}%)")

    print(f"{'='*80}\n")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Main execution pipeline"""

    print("=" * 80)
    print("SERVICE PUBLIC SCRAPER → MONGODB ATLAS [PARALLEL MODE]")
    print(f"Collection: RUCHE_datalake.{COLLECTION_NAME}")
    print("=" * 80)

    # Check credentials
    if not MISTRAL_API_KEY:
        print("[ERROR] MISTRAL_API_KEY not found in .env")
        return

    # Connect to MongoDB
    print("\n[STEP 1/3] Connecting to MongoDB Atlas")
    collection = get_collection(COLLECTION_NAME)
    if collection is None:
        print("[ERROR] Could not connect to MongoDB")
        return

    # Create unique index
    create_unique_index(collection, "id")

    # Show current stats
    current_count = count_documents(collection)
    print(f"[DB] Current documents in collection: {current_count}")

    # Scrape with parallel processing
    print("\n[STEP 2/3] Scraping 'data' keyword (PARALLEL)")
    scrape_keyword_parallel("data", collection, workers=4)

    print("\n[STEP 3/3] Scraping 'ia' keyword (PARALLEL)")
    scrape_keyword_parallel("ia", collection, workers=4)

    # Final stats
    final_count = count_documents(collection)
    new_docs = final_count - current_count

    print(f"\n{'='*80}")
    print("SCRAPING COMPLETE")
    print("=" * 80)
    print(f"Collection before: {current_count} documents")
    print(f"Collection after: {final_count} documents")
    print(f"New documents added: {new_docs}")

    stats = get_collection_stats(collection)
    print("\n[DB] Collection statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n[SUCCESS] Check your MongoDB Atlas dashboard")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WARNING] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()
    finally:
        input("\nPress ENTER to exit...")
