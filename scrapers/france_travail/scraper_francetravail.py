"""
France Travail API Scraper
Scrapes job offers containing "data" or "AI" keywords
Stores results in MongoDB Atlas (RUCHE_datalake.francetravail_raw)
"""

import os
import time
import re
import threading
from collections import deque
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
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
CLIENT_ID = os.getenv("FT_CLIENT_ID")
CLIENT_SECRET = os.getenv("FT_CLIENT_SECRET")

# API URLs
AUTH_URL = (
    "https://entreprise.pole-emploi.fr/connexion/oauth2/access_token?realm=/partenaire"
)
API_SEARCH_URL = (
    "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
)
API_DETAIL_URL = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres"

# Collection name for France Travail
COLLECTION_NAME = "francetravail_raw"


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class JobOffer:
    """France Travail job offer with all details"""

    id: str
    intitule: str
    description: str
    lieu: str
    type_contrat: str
    date_creation: str
    date_actualisation: str
    entreprise: Optional[str] = None
    salaire: Optional[str] = None
    experience: Optional[str] = None
    competences: List[str] = field(default_factory=list)
    formations: List[str] = field(default_factory=list)
    langues: List[str] = field(default_factory=list)
    permis: List[str] = field(default_factory=list)
    url_origine: Optional[str] = None
    matched_keywords: List[str] = field(default_factory=list)
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB"""
        return asdict(self)


# =============================================================================
# RATE LIMITER
# =============================================================================


class RateLimiter:
    """Thread-safe rate limiter for API calls"""

    def __init__(self, max_calls: int = 9):
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
# KEYWORD MATCHING
# =============================================================================

KEYWORD_PATTERNS = {
    "data": r"\bdata\b",
    "données": r"\bdonn[ée]es?\b",
    "IA": r"\bia\b",
    "intelligence artificielle": r"\bintelligence\s+artificielle\b",
    "AI": r"\bai\b",
}


def matches_keywords(text: str) -> tuple[bool, List[str]]:
    """Check if text contains target keywords"""
    text_lower = text.lower()
    matched = [
        kw for kw, pattern in KEYWORD_PATTERNS.items() if re.search(pattern, text_lower)
    ]
    return (len(matched) > 0, matched)


# =============================================================================
# API FUNCTIONS
# =============================================================================


def get_token(limiter: RateLimiter) -> Optional[str]:
    """Get OAuth2 access token"""
    limiter.wait()
    try:
        response = requests.post(
            AUTH_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "scope": "api_offresdemploiv2 o2dsoffre",
            },
            timeout=10,
        )

        if response.status_code == 200:
            print("[AUTH] Token obtained")
            return response.json().get("access_token")
        print(f"[AUTH ERROR] {response.status_code}")
    except Exception as e:
        print(f"[AUTH ERROR] {e}")
    return None


def parse_offer_summary(
    data: dict,
) -> tuple[str, str, str, str, Optional[str], Optional[str]]:
    """Parse offer summary from API response"""
    lieu = data.get("lieuTravail", {}).get("libelle", "Not specified")
    entreprise = data.get("entreprise", {}).get("nom")
    salaire = data.get("salaire", {}).get("libelle")
    return (
        data.get("id", ""),
        data.get("intitule", ""),
        lieu,
        data.get("typeContratLibelle", "Not specified"),
        entreprise,
        salaire,
    )


def search_offers(
    token: str,
    limiter: RateLimiter,
    keyword: str,
    seen_ids: Set[str],
    max_results: Optional[int] = None,
) -> List[tuple]:
    """Search all job offers with pagination and deduplication"""
    print(f"\n{'='*80}\n[SEARCH] Keywords: '{keyword}'\n{'='*80}")

    offers = []
    start = 0
    batch = 150

    while True:
        if max_results and start >= max_results:
            break

        limiter.wait()
        end = (
            min(start + batch - 1, max_results - 1)
            if max_results
            else start + batch - 1
        )

        try:
            response = requests.get(
                API_SEARCH_URL,
                params={"motsCles": keyword, "range": f"{start}-{end}"},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
                timeout=15,
            )

            if response.status_code not in (200, 206):
                print(f"[SEARCH] Status {response.status_code}")
                break

            results = response.json().get("resultats", [])
            if not results:
                break

            batch_new = 0
            for offre in results:
                offer_id = offre.get("id", "")
                if offer_id not in seen_ids:
                    seen_ids.add(offer_id)
                    offers.append(parse_offer_summary(offre))
                    batch_new += 1

            content_range = response.headers.get("Content-Range", "?")
            print(
                f"[SEARCH] Range {start}-{end} (Total: {content_range}): {len(results)} offers, {batch_new} new"
            )

            if len(results) < batch:
                break
            start += batch

        except Exception as e:
            print(f"[SEARCH ERROR] {e}")
            break

    print(f"[SEARCH] Complete: {len(offers)} unique offers\n{'='*80}\n")
    return offers


def get_offer_detail(
    token: str, offer_id: str, limiter: RateLimiter
) -> Optional[JobOffer]:
    """Fetch complete job offer details"""
    limiter.wait()
    try:
        response = requests.get(
            f"{API_DETAIL_URL}/{offer_id}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=15,
        )

        if response.status_code != 200:
            return None

        data = response.json()

        lieu = data.get("lieuTravail", {}).get("libelle", "Not specified")
        entreprise = data.get("entreprise", {}).get("nom")
        salaire = data.get("salaire", {}).get("libelle")
        url = data.get("origineOffre", {}).get("urlOrigine")

        competences = [c.get("libelle", "") for c in data.get("competences", [])]
        formations = [f.get("niveauLibelle", "") for f in data.get("formations", [])]
        langues = [l.get("libelle", "") for l in data.get("langues", [])]
        permis = [p.get("libelle", "") for p in data.get("permis", [])]

        intitule = data.get("intitule", "")
        description = data.get("description", "")

        full_text = f"{intitule} {description} {' '.join(competences)}"
        _, matched = matches_keywords(full_text)

        return JobOffer(
            id=data.get("id", ""),
            intitule=intitule,
            description=description,
            lieu=lieu,
            type_contrat=data.get("typeContratLibelle", "Not specified"),
            date_creation=data.get("dateCreation", ""),
            date_actualisation=data.get("dateActualisation", ""),
            entreprise=entreprise,
            salaire=salaire,
            experience=data.get("experienceLibelle"),
            competences=competences,
            formations=formations,
            langues=langues,
            permis=permis,
            url_origine=url,
            matched_keywords=matched,
        )
    except:
        return None


# =============================================================================
# SCRAPING FUNCTIONS
# =============================================================================


def scrape_to_mongo(
    token: str,
    summaries: List[tuple],
    limiter: RateLimiter,
    collection,
    workers: int = 8,
    verify: bool = False,
):
    """Scrape offers in parallel and push to MongoDB"""

    print(
        f"\n{'='*80}\n[SCRAPE] Processing {len(summaries)} offers -> MongoDB\n{'='*80}"
    )

    lock = threading.Lock()
    stats = {"ok": 0, "filtered": 0, "failed": 0, "db_ops": 0}
    buffer = []
    start_time = time.time()

    def process_one(summary: tuple) -> Optional[JobOffer]:
        """Process single offer"""
        offer = get_offer_detail(token, summary[0], limiter)
        if not offer:
            return None
        if verify:
            has_match, _ = matches_keywords(f"{offer.intitule} {offer.description}")
            if not has_match:
                return None
        return offer

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, s): s for s in summaries}

        for i, future in enumerate(as_completed(futures), 1):
            try:
                offer = future.result()

                with lock:
                    if offer:
                        buffer.append(offer.to_dict())
                        stats["ok"] += 1

                        if len(buffer) >= 20:
                            ops_count = bulk_upsert(collection, buffer)
                            stats["db_ops"] += ops_count
                            buffer.clear()

                    else:
                        stats["filtered" if verify else "failed"] += 1

                if i % 10 == 0 or i == len(summaries):
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    pct = i / len(summaries) * 100
                    print(
                        f"\r[SCRAPE] {pct:.0f}% | {i}/{len(summaries)} | "
                        f"Queued:{stats['ok']} Filtered:{stats['filtered']} | "
                        f"DB Ops:{stats['db_ops']} | {rate:.1f}/s",
                        end="",
                        flush=True,
                    )

            except Exception as e:
                stats["failed"] += 1
                print(f"\n[ERROR] {e}")

    if buffer:
        print("\n[DB] Flushing remaining buffer...")
        ops_count = bulk_upsert(collection, buffer)
        stats["db_ops"] += ops_count

    print(f"\n[SCRAPE] Complete: {stats['ok']} offers processed -> MongoDB\n{'='*80}\n")
    return stats


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Main execution pipeline"""

    print("=" * 80)
    print("FRANCE TRAVAIL SCRAPER -> MONGODB ATLAS")
    print(f"Collection: RUCHE_datalake.{COLLECTION_NAME}")
    print("=" * 80)

    # Check credentials
    if not CLIENT_ID or not CLIENT_SECRET:
        print("[ERROR] Missing FT_CLIENT_ID and FT_CLIENT_SECRET in .env")
        return

    # Step 0: Connect to MongoDB
    print("\n[STEP 0/3] Connecting to MongoDB Atlas")
    collection = get_collection(COLLECTION_NAME)
    if collection is None:
        print("[ERROR] Could not connect to MongoDB")
        return

    # Create unique index on 'id' field
    create_unique_index(collection, "id")

    # Show current collection stats
    current_count = count_documents(collection)
    print(f"[DB] Current documents in collection: {current_count}")

    # Initialize
    limiter = RateLimiter(max_calls=9)
    seen_ids: Set[str] = set()

    # Step 1: Authentication
    print("\n[STEP 1/3] Authentication")
    token = get_token(limiter)
    if not token:
        print("[ERROR] Authentication failed")
        return

    # Step 2: Search
    print("\n[STEP 2/3] Searching offers")
    summaries = search_offers(
        token=token,
        limiter=limiter,
        keyword="data",  # ← CORRIGÉ : "data" seul
        seen_ids=seen_ids,
        max_results=None,
    )

    if not summaries:
        print("[ERROR] No offers found")
        return

    print(f"\n[INFO] Found {len(summaries)} unique offers")

    # Confirmation
    if len(summaries) > 100:
        est_time = len(summaries) / 9
        print(f"[WARNING] About to scrape {len(summaries)} offers")
        print(f"[INFO] Estimated time: ~{est_time:.0f}s ({est_time/60:.1f} min)")
        response = input("Continue? [Y/n]: ")
        if response.lower() == "n":
            print("[INFO] Aborted by user")
            return

    # Step 3: Scrape & Upsert
    print("\n[STEP 3/3] Scraping details & Upserting to MongoDB")

    scrape_stats = scrape_to_mongo(
        token=token,
        summaries=summaries,
        limiter=limiter,
        collection=collection,
        workers=8,
        verify=False,  # ← CORRIGÉ : False (pas de double filtrage)
    )

    # Final statistics
    final_count = count_documents(collection)
    new_docs = final_count - current_count

    print(f"\n{'='*80}")
    print("SCRAPING COMPLETE")
    print("=" * 80)
    print(f"Searched: {len(summaries)} offers")
    print(f"Processed: {scrape_stats['ok']} offers")
    print(f"Filtered: {scrape_stats['filtered']} offers")
    print(f"Failed: {scrape_stats['failed']} offers")
    print(f"DB Operations: {scrape_stats['db_ops']}")
    print(f"Collection before: {current_count} documents")
    print(f"Collection after: {final_count} documents")
    print(f"New documents added: {new_docs}")
    print(f"API calls: {limiter.stats()}")
    print("=" * 80)

    # Show collection stats
    print("\n[DB] Final collection statistics:")
    stats = get_collection_stats(collection)
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
