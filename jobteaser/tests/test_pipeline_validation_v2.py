"""
Test de validation PIPELINE JobTeaser — V2 (après correction navigation)
Objectif : valider scraping + NLP sur un petit échantillon
"""

import os
import json
import time
import sys
from datetime import datetime
from collections import Counter
from pprint import pprint

import config
from scraper import run_scraper
from nlp import enrich_offers_jobteaser


# ============================================================
# CONFIG TEST (override temporaire)
# ============================================================

TEST_KEYWORDS = ["data engineer", "data analyst"]
TEST_MAX_LIST_PAGES = 3  # 🔑 2 pages liste seulement
TEST_MAX_PER_KEYWORD = 25  # permet > 1 détail / page
TEST_MAX_TOTAL = 40

# backup config
BK = {
    "SEARCH_QUERIES": config.SEARCH_QUERIES,
    "MAX_DETAIL_PAGES_TOTAL": config.MAX_DETAIL_PAGES_TOTAL,
    "MAX_DETAIL_PAGES_PER_KEYWORD": config.MAX_DETAIL_PAGES_PER_KEYWORD,
    "MAX_LIST_PAGES_PER_KEYWORD": getattr(config, "MAX_LIST_PAGES_PER_KEYWORD", None),
}


def fail(msg):
    print(f"\n❌ TEST FAILED — {msg}")
    sys.exit(1)


def main():
    print("🧪 TEST PIPELINE JOBTEASER — VALIDATION V2")
    print("=" * 70)

    try:
        # override config
        config.SEARCH_QUERIES = TEST_KEYWORDS
        config.MAX_DETAIL_PAGES_TOTAL = TEST_MAX_TOTAL
        config.MAX_DETAIL_PAGES_PER_KEYWORD = TEST_MAX_PER_KEYWORD
        config.MAX_LIST_PAGES_PER_KEYWORD = TEST_MAX_LIST_PAGES

        print("🔧 CONFIG TEST")
        print(f"- keywords              : {TEST_KEYWORDS}")
        print(f"- max list pages        : {TEST_MAX_LIST_PAGES}")
        print(f"- max detail / keyword  : {TEST_MAX_PER_KEYWORD}")
        print(f"- max detail total      : {TEST_MAX_TOTAL}")
        print("- IGNORE_UNKNOWN_TIME   :", config.IGNORE_UNKNOWN_TIME)

        # ====================================================
        # 1️⃣ SCRAPING
        # ====================================================
        t0 = time.perf_counter()
        raw_offers, stats = run_scraper()
        t_scrape = time.perf_counter() - t0

        print(f"\n✔ Scraping terminé : {len(raw_offers)} offres | {t_scrape:.1f}s")
        pprint(stats)

        if len(raw_offers) == 0:
            fail("0 offre scrapée — régression scraping")

        # 🔑 ASSERT CLÉ : plus d'offres que de pages
        if len(raw_offers) <= TEST_MAX_LIST_PAGES:
            fail(
                "Le nombre d'offres finales est ≤ au nombre de pages liste "
                "(la correction navigation ne fonctionne pas)"
            )

        # ====================================================
        # 2️⃣ CHECK RAW
        # ====================================================
        ids = [o.get("id") for o in raw_offers]
        id_counts = Counter(ids)

        if not all(ids):
            fail("Certaines offres RAW n'ont pas d'id")

        if not all(v == 1 for v in id_counts.values()):
            fail("Doublons UUID détectés dans RAW")

        sample = raw_offers[0]
        for field in ["id", "url", "title", "company", "description_raw"]:
            if not sample.get(field):
                fail(f"Champ RAW manquant ou vide : {field}")

        print("✔ RAW OK (structure + unicité)")

        # ====================================================
        # 3️⃣ NLP
        # ====================================================
        t0 = time.perf_counter()
        enriched = enrich_offers_jobteaser(raw_offers)
        t_nlp = time.perf_counter() - t0

        if len(enriched) != len(raw_offers):
            fail("Perte d'offres entre RAW et ENRICHED")

        enriched_sample = enriched[0]
        for field in [
            "description_clean",
            "hard_skills",
            "soft_skills",
            "role_family",
            "nlp_success",
            "publication_date",
        ]:
            if field not in enriched_sample:
                fail(f"Champ NLP manquant : {field}")

        if "description_raw" in enriched_sample:
            fail("description_raw ne doit pas être présent dans ENRICHED")

        print(f"✔ NLP OK : {len(enriched)} offres | {t_nlp:.2f}s")

        # ====================================================
        # 4️⃣ OUTPUT FILES
        # ====================================================
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        raw_path = f"{config.OUTPUT_DIR}/jobteaser_raw_VALIDATION_{ts}.json"
        enriched_path = f"{config.OUTPUT_DIR}/jobteaser_enriched_VALIDATION_{ts}.json"

        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_offers, f, ensure_ascii=False, indent=2)

        with open(enriched_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)

        print("\n📦 FICHIERS GÉNÉRÉS")
        print(f"- RAW      : {raw_path}")
        print(f"- ENRICHED : {enriched_path}")

        # ====================================================
        # 5️⃣ RÉSUMÉ FINAL
        # ====================================================
        print("\n🎉 TEST PIPELINE VALIDÉ (V2)")
        print("=" * 70)
        print(f"- pages liste testées : {TEST_MAX_LIST_PAGES}")
        print(f"- offres finales     : {len(enriched)}")
        print("👉 La correction navigation est VALIDÉE.")

    finally:
        # restore config
        config.SEARCH_QUERIES = BK["SEARCH_QUERIES"]
        config.MAX_DETAIL_PAGES_TOTAL = BK["MAX_DETAIL_PAGES_TOTAL"]
        config.MAX_DETAIL_PAGES_PER_KEYWORD = BK["MAX_DETAIL_PAGES_PER_KEYWORD"]
        if BK["MAX_LIST_PAGES_PER_KEYWORD"] is None:
            if hasattr(config, "MAX_LIST_PAGES_PER_KEYWORD"):
                delattr(config, "MAX_LIST_PAGES_PER_KEYWORD")
        else:
            config.MAX_LIST_PAGES_PER_KEYWORD = BK["MAX_LIST_PAGES_PER_KEYWORD"]


if __name__ == "__main__":
    main()
