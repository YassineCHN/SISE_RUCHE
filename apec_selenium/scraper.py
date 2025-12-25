"""
Module de scraping APEC avec Selenium.
Gère la collecte des URLs et le scraping des offres.
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
from typing import List, Optional, Dict
from config import (
    CHROME_OPTIONS,
    APEC_SEARCH_URL,
    APEC_BASE_URL,
    APEC_DATE_FILTER,
    PAGE_LOAD_DELAY,
    SCRAPING_DELAY,
    MAX_EMPTY_PAGES,
    MAX_OFFERS_PER_KEYWORD,
)


def create_chrome_driver(for_scraping: bool = False) -> webdriver.Chrome:
    """
    Crée un driver Chrome avec les options optimisées.
    
    Args:
        for_scraping: Si True, ajoute des optimisations pour le scraping
        
    Returns:
        Instance de ChromeDriver
    """
    options = Options()
    for opt in CHROME_OPTIONS:
        options.add_argument(opt)
    
    # Options supplémentaires pour le scraping
    if for_scraping:
        options.add_argument("--window-size=1280,720")
    
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )


def find_offers_for_keyword(keyword: str, driver: webdriver.Chrome) -> List[str]:
    """
    Trouve toutes les URLs d'offres pour un mot-clé.
    
    Args:
        keyword: Mot-clé de recherche
        driver: Instance de ChromeDriver
        
    Returns:
        Liste d'URLs d'offres
    """
    print(f"\n🔍 Recherche: '{keyword}'")
    
    urls = set()
    page = 0
    consecutive_empty = 0
    
    while consecutive_empty < MAX_EMPTY_PAGES and len(urls) < MAX_OFFERS_PER_KEYWORD:
        # Construire l'URL avec filtre < 30 jours
        search_url = (
            f"{APEC_SEARCH_URL}"
            f"?motsCles={keyword.replace(' ', '+')}"
            f"&anciennetePublication={APEC_DATE_FILTER}"
            f"&page={page}"
        )
        
        driver.get(search_url)
        time.sleep(PAGE_LOAD_DELAY)
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        page_count = 0
        
        # Extraire les URLs
        for link in soup.find_all("a", href=True):
            if 'detail-offre' in link['href']:
                href = link['href']
                
                # Construire l'URL complète
                full_url = href if href.startswith('http') else f"{APEC_BASE_URL}{href}"
                base_url = full_url.split('?')[0]
                
                if base_url not in urls:
                    urls.add(base_url)
                    page_count += 1
                    
                    # Limite par mot-clé
                    if len(urls) >= MAX_OFFERS_PER_KEYWORD:
                        break
        
        # Gestion des pages vides
        if page_count > 0:
            consecutive_empty = 0
            print(f"   Page {page+1:>2}: +{page_count:>2} offres | Total: {len(urls):>4}")
        else:
            consecutive_empty += 1
        
        page += 1
    
    print(f"   ✅ {len(urls)} offres trouvées")
    return list(urls)


def collect_all_urls(keywords: List[str]) -> List[str]:
    """
    Collecte toutes les URLs pour une liste de mots-clés.
    
    Args:
        keywords: Liste de mots-clés
        
    Returns:
        Liste d'URLs uniques
    """
    driver = None
    all_urls = set()
    
    try:
        driver = create_chrome_driver()
        
        for idx, keyword in enumerate(keywords, 1):
            print(f"\n[{idx}/{len(keywords)}]", end=" ")
            urls = find_offers_for_keyword(keyword, driver)
            all_urls.update(urls)
        
        print(f"\n📊 TOTAL: {len(all_urls)} offres uniques (max {MAX_OFFERS_PER_KEYWORD} par mot-clé)\n")
        return list(all_urls)
        
    finally:
        if driver:
            driver.quit()


def scrape_offer(url: str) -> Optional[Dict]:
    """
    Scrape une offre APEC et récupère le HTML brut.
    
    Args:
        url: URL de l'offre
        
    Returns:
        Dictionnaire avec le contenu brut ou None si échec
    """
    driver = None
    
    try:
        driver = create_chrome_driver(for_scraping=True)
        
        driver.get(url)
        time.sleep(SCRAPING_DELAY)
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # Vérification rapide
        if len(soup.get_text()) < 500:
            return None
        
        # Extraction du titre
        title = None
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
        
        # Extraction du contenu
        main = soup.find("main")
        if not main:
            return None
        
        # Nettoyage
        for elem in main.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            elem.decompose()
        
        content = main.get_text(separator="\n", strip=True)
        
        if len(content) < 200:
            return None
        
        return {
            "url": url,
            "title": title,
            "description_raw": content,
            "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        return None
        
    finally:
        if driver:
            driver.quit()