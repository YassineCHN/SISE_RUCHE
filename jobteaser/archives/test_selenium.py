from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def test_selenium_debug():
    print("🔍 Testing JobTeaser with Selenium (DEBUG MODE)...")
    
    # Configuration Chrome
    chrome_options = Options()
    # chrome_options.add_argument('--headless')  # ← DÉSACTIVÉ pour voir la fenêtre
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Lancer Chrome
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )
    
    try:
        # Accéder à JobTeaser
        url = "https://www.jobteaser.com/fr/job-offers?q=data&page=1"
        print(f"   → Accessing {url}")
        driver.get(url)
        
        # ATTENDRE PLUS LONGTEMPS pour le chargement JS
        print("   → Waiting for page to load...")
        time.sleep(5)  # 5 secondes
        
        # Vérifier le titre
        print(f"   → Page title: {driver.title}")
        
        # Sauvegarder le HTML pour inspection
        html_content = driver.page_source
        with open('jobteaser_page.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        print("   → HTML saved to jobteaser_page.html")
        
        # Vérifier si bloqué
        if "403" in driver.title or "Forbidden" in html_content[:1000]:
            print("❌ Still blocked (403)")
            return False
        
        if "Cloudflare" in html_content[:5000]:
            print("⚠️  Cloudflare challenge detected")
            print("   Waiting 10 more seconds for challenge to complete...")
            time.sleep(10)
            html_content = driver.page_source
        
        # Essayer différents sélecteurs
        selectors_to_try = [
            ("CLASS_NAME", "JobAdCard_main__1mTeA"),
            ("CSS_SELECTOR", "div[data-testid='jobad-card']"),
            ("CSS_SELECTOR", ".JobAdCard_main__1mTeA"),
            ("TAG_NAME", "article"),
            ("CSS_SELECTOR", "li > div"),
        ]
        
        for method, selector in selectors_to_try:
            try:
                print(f"   → Trying selector: {method} = '{selector}'")
                if method == "CLASS_NAME":
                    elements = driver.find_elements(By.CLASS_NAME, selector)
                elif method == "CSS_SELECTOR":
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                elif method == "TAG_NAME":
                    elements = driver.find_elements(By.TAG_NAME, selector)
                
                if elements:
                    print(f"   ✅ Found {len(elements)} elements with {selector}")
                    if len(elements) > 0:
                        print(f"   First element text: {elements[0].text[:200]}...")
                else:
                    print(f"   ❌ No elements found with {selector}")
            except Exception as e:
                print(f"   ❌ Error with {selector}: {e}")
        
        # Compter tous les divs (debug)
        all_divs = driver.find_elements(By.TAG_NAME, "div")
        print(f"\n   → Total <div> elements on page: {len(all_divs)}")
        
        # Chercher "data scientist" dans le texte
        if "data scientist" in html_content.lower():
            print("   ✅ Found 'data scientist' in page content")
        else:
            print("   ❌ No 'data scientist' found in page")
        
        print("\n   → Press ENTER to close browser...")
        input()
        
    finally:
        driver.quit()

if __name__ == "__main__":
    test_selenium_debug()