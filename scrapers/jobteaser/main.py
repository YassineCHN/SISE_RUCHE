import os
import json
from datetime import datetime

from scraper import run_scraper
from nlp import enrich_offers_jobteaser
from config import OUTPUT_DIR


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1️⃣ SCRAPING
    raw_offers, stats = run_scraper()

    raw_path = f"{OUTPUT_DIR}/jobteaser_raw_{ts}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_offers, f, ensure_ascii=False, indent=2)

    # 2️⃣ NLP
    enriched = enrich_offers_jobteaser(raw_offers)

    enriched_path = f"{OUTPUT_DIR}/jobteaser_enriched_{ts}.json"
    with open(enriched_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    print("✅ Pipeline terminé")
    print(f"- RAW      : {raw_path}")
    print(f"- ENRICHED : {enriched_path}")
    print(f"- Offres   : {len(enriched)}")


if __name__ == "__main__":
    main()
