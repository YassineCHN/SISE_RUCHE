"""
Test de validation pipeline JobTeaser
À lancer AVANT le gros scraping initial
"""

import sys
from collections import Counter
from pprint import pprint

import config
from scraper import run_scraper
from nlp import enrich_offers_jobteaser


# ============================================================
# CONFIG DE TEST (override temporaire)
# ============================================================

TEST_LIMIT_PER_KEYWORD = 3
TEST_TOTAL_LIMIT = 10

config.MAX_DETAIL_PAGES_PER_KEYWORD = TEST_LIMIT_PER_KEYWORD
config.MAX_DETAIL_PAGES_TOTAL = TEST_TOTAL_LIMIT
config.MAX_LIST_AGE_DAYS = 30
config.IGNORE_UNKNOWN_TIME = True


# ============================================================
# HELPERS ASSERT
# ============================================================


def assert_true(cond, msg):
    if not cond:
        print(f"❌ TEST FAILED: {msg}")
        sys.exit(1)


def assert_not_empty(val, msg):
    assert_true(val is not None and val != "", msg)


# ============================================================
# TEST PIPELINE
# ============================================================


def main():
    print("🧪 TEST PIPELINE JOBTEASER — AVANT GROS SCRAPING")
    print("=" * 60)

    # 1️⃣ SCRAPING
    raw_offers, stats = run_scraper()

    print("\n📊 Stats scraping par keyword :")
    pprint(stats)

    assert_true(len(raw_offers) > 0, "Aucune offre scrapée")

    # 2️⃣ CHECK RAW
    ids = [o.get("id") for o in raw_offers]
    id_counts = Counter(ids)

    assert_true(all(ids), "Certaines offres n'ont pas d'id")
    assert_true(
        all(v == 1 for v in id_counts.values()),
        "Doublons détectés dans les UUID",
    )

    sample = raw_offers[0]
    required_raw_fields = [
        "id",
        "url",
        "title",
        "company",
        "description_raw",
        "publication_date_raw",
    ]

    for f in required_raw_fields:
        assert_not_empty(sample.get(f), f"Champ RAW manquant ou vide: {f}")

    print(f"\n✅ RAW OK — {len(raw_offers)} offres uniques")

    # 3️⃣ NLP
    enriched = enrich_offers_jobteaser(raw_offers)
    assert_true(len(enriched) == len(raw_offers), "Perte d'offres au NLP")

    enriched_sample = enriched[0]

    # champs clés NLP
    required_enriched_fields = [
        "description_clean",
        "hard_skills",
        "soft_skills",
        "role_family",
        "nlp_success",
        "publication_date",
    ]

    for f in required_enriched_fields:
        assert_true(f in enriched_sample, f"Champ NLP manquant: {f}")

    # règles métier
    assert_true(
        "description_raw" not in enriched_sample,
        "description_raw ne doit pas être dans ENRICHED",
    )

    print("✅ NLP OK")

    # 4️⃣ CHECK DATE
    parsed_dates = [
        o["publication_date"] for o in enriched if o.get("publication_date")
    ]

    print(f"📅 Dates publiées parsées : {len(parsed_dates)}/{len(enriched)}")

    # 5️⃣ RÉSUMÉ FINAL
    print("\n🎉 TEST PIPELINE RÉUSSI")
    print("=" * 60)
    print(f"- Offres scrapées   : {len(raw_offers)}")
    print(f"- Offres enrichies  : {len(enriched)}")
    print("- IGNORE_UNKNOWN_TIME :", config.IGNORE_UNKNOWN_TIME)
    print("- MAX_DETAIL_TOTAL   :", config.MAX_DETAIL_PAGES_TOTAL)

    print("\n👉 Tu peux lancer le gros scraping en confiance.")


if __name__ == "__main__":
    main()
