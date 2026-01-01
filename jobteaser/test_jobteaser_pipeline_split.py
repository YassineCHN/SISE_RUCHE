# test_jobteaser_pipeline_split.py

import os
import json
import time
from datetime import datetime

import config
from scraper import run_scraper
from nlp import enrich_offers_jobteaser


TEST_KEYWORDS = [
    "data engineer",
    "data scientist",
    # "data analyst",
    # "machine learning",
    # "intelligence artificielle",
]
TEST_MAX_TOTAL = 6
TEST_MAX_PER_KEYWORD = 1


def main():
    print("🧪 TEST PIPELINE JOBTEASER — VERSION DÉCOUPÉE")
    print(f"- IGNORE_UNKNOWN_TIME = {config.IGNORE_UNKNOWN_TIME}")
    print("- Dédoublonnage global actif\n")

    # backup config
    bk_queries = config.SEARCH_QUERIES
    bk_total = config.MAX_DETAIL_PAGES_TOTAL
    bk_kw = config.MAX_DETAIL_PAGES_PER_KEYWORD

    try:
        # override
        config.SEARCH_QUERIES = TEST_KEYWORDS
        config.MAX_DETAIL_PAGES_TOTAL = TEST_MAX_TOTAL
        config.MAX_DETAIL_PAGES_PER_KEYWORD = TEST_MAX_PER_KEYWORD

        t0 = time.perf_counter()
        raw_offers, stats = run_scraper()
        t_scrape = time.perf_counter() - t0

        print(f"\n✔ Scraping terminé : {len(raw_offers)} offres | {t_scrape:.1f}s")

        # FAIL FAST
        if len(raw_offers) == 0:
            print(
                "\n❌ ERREUR : 0 offre collectée. Probable régression (sélecteurs / filtres / arrêt)."
            )
            print("📊 Stats par mot-clé :")
            for kw, s in stats.items():
                print(f"- {kw}: {s}")
            raise SystemExit(2)

        t0 = time.perf_counter()
        enriched = enrich_offers_jobteaser(raw_offers)
        t_nlp = time.perf_counter() - t0
        print(f"✔ NLP terminé : {len(enriched)} offres | {t_nlp:.2f}s")

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        raw_path = f"{config.OUTPUT_DIR}/jobteaser_raw_TEST_{ts}.json"
        enriched_path = f"{config.OUTPUT_DIR}/jobteaser_enriched_TEST_{ts}.json"

        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_offers, f, ensure_ascii=False, indent=2)

        with open(enriched_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)

        print("\n📦 FICHIERS GÉNÉRÉS")
        print(f"- RAW      : {raw_path}")
        print(f"- ENRICHED : {enriched_path}")

        print("\n📊 STATS PAR MOT-CLÉ")
        for kw, s in stats.items():
            print(
                f"- {kw}: seen={s['cards_seen']} | dup={s['duplicates']} | "
                f"unk={s['ignored_unknown_time']} | old={s['ignored_too_old']} | "
                f"filtered={s['filtered_title']} | opened={s['detail_opened']} | kept={s['kept']}"
            )

        print("\n✅ TEST OK")

    finally:
        # restore
        config.SEARCH_QUERIES = bk_queries
        config.MAX_DETAIL_PAGES_TOTAL = bk_total
        config.MAX_DETAIL_PAGES_PER_KEYWORD = bk_kw


if __name__ == "__main__":
    main()
