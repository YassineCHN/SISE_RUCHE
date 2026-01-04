"""
Test réduit du scraper APRÈS correction stratégie A
"""

import config
from scraper import run_scraper

# 🔧 paramètres de test
config.SEARCH_QUERIES = ["data engineer"]
config.MAX_DETAIL_PAGES_PER_KEYWORD = 10
config.MAX_DETAIL_PAGES_TOTAL = 10
config.MAX_LIST_AGE_DAYS = 30
config.IGNORE_UNKNOWN_TIME = True


def main():
    print("🧪 TEST SCRAPER APRÈS FIX (STRATÉGIE A)")
    print("=" * 60)

    results, stats = run_scraper()

    print("\n📊 Stats :")
    for k, v in stats.items():
        print(k, v)

    print("\n📦 Résultats finaux :", len(results))
    for i, r in enumerate(results[:5], 1):
        print(f"{i}. {r.get('title')}")


if __name__ == "__main__":
    main()
