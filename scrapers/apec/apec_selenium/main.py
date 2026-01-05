"""
Script principal du scraper APEC.
Orchestre la collecte, le scraping et l'extraction NLP.
"""

import json
import time
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from config import MAX_WORKERS_SCRAPING, OUTPUT_DIR, FILE_ENCODING, JSON_INDENT
from scraper import collect_all_urls, scrape_offer
from nlp import enrich_offers
from utils import (
    ProgressTracker,
    print_header,
    print_section,
    get_user_keywords,
    get_num_workers,
    display_summary,
)

# Lock pour thread-safety
progress_lock = Lock()


def parallel_scraping(urls: list, num_workers: int) -> list:
    """
    Scrape les offres en parallèle.
    
    Args:
        urls: Liste d'URLs à scraper
        num_workers: Nombre de threads
        
    Returns:
        Liste des offres brutes scrapées
    """
    tracker = ProgressTracker(len(urls))
    raw_offers = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(scrape_offer, url): url for url in urls}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                
                with progress_lock:
                    if result:
                        tracker.increment_success()
                        raw_offers.append(result)
                    else:
                        tracker.increment_failed()
                    
                    tracker.print_progress()
                    
            except Exception as e:
                with progress_lock:
                    tracker.increment_failed()
                    tracker.print_progress()
    
    print()  # Nouvelle ligne après la barre
    tracker.print_final_stats()
    
    return raw_offers, tracker


def save_results(offers: list, keywords: list) -> str:
    """
    Sauvegarde les résultats dans un fichier JSON.
    
    Args:
        offers: Liste des offres enrichies
        keywords: Mots-clés utilisés
        
    Returns:
        Nom du fichier créé
    """
    # Créer le dossier de sortie si nécessaire
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Filtrer et nettoyer les mots-clés (enlever chemins, caractères invalides)
    clean_keywords = []
    for k in keywords:
        # Ignorer les chemins de fichiers
        if '\\' in k or '/' in k or ':' in k:
            continue
        # Nettoyer les caractères invalides
        clean_k = re.sub(r'[<>:"/\\|?*]', '', k)
        clean_k = clean_k.replace(" ", "")[:15]  # Max 15 caractères
        if clean_k:
            clean_keywords.append(clean_k)
    
    # Nom du fichier
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    keywords_str = "_".join(clean_keywords[:3]) if clean_keywords else "scraping"
    filename = f"apec_{keywords_str}_{len(offers)}offres_{timestamp}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Sauvegarde
    with open(filepath, "w", encoding=FILE_ENCODING) as f:
        json.dump(offers, f, ensure_ascii=False, indent=JSON_INDENT)
    
    return filepath


def main():
    """Fonction principale."""
    
    # En-tête
    print_header()
    
    # Configuration
    keywords = get_user_keywords()
    num_workers = get_num_workers()
    
    # ÉTAPE 1: Collecte des URLs
    print_section("ÉTAPE 1: Collecte des URLs")
    urls = collect_all_urls(keywords)
    
    if not urls:
        print("\n❌ Aucune URL trouvée!")
        return
    
    # ÉTAPE 2: Scraping parallèle
    print_section(f"ÉTAPE 2: Scraping parallèle ({num_workers} threads)")
    raw_offers, tracker = parallel_scraping(urls, num_workers)
    
    if not raw_offers:
        print("\n❌ Aucune offre scrapée avec succès!")
        return
    
    # ÉTAPE 3: NLP
    print_section("ÉTAPE 3: Application du NLP")
    enriched_offers = enrich_offers(raw_offers)
    
    # Sauvegarde
    filepath = save_results(enriched_offers, keywords)
    
    # Résumé final
    print()
    display_summary(enriched_offers, tracker)
    
    print()
    print("=" * 90)
    print(f"📁 Fichier créé: {filepath}")
    print("=" * 90)
    print()
    print(f"✅ {len(enriched_offers)} offres < 30 jours avec NLP enrichi!")
    print()


if __name__ == "__main__":
    main()