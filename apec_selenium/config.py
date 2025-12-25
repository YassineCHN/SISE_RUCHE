"""
Configuration centralisée du scraper APEC.
"""

# ==================== PARAMÈTRES DE SCRAPING ====================

# Nombre maximum d'offres par mot-clé
MAX_OFFERS_PER_KEYWORD = 600

# Nombre de threads pour le scraping parallèle
MAX_WORKERS_SCRAPING = 10  # Recommandé: 10-15

# Délais (en secondes)
SCRAPING_DELAY = 2.0  # Temps d'attente après chargement d'une offre
PAGE_LOAD_DELAY = 2.0  # Temps d'attente après chargement page de recherche

# Nombre de pages vides consécutives avant arrêt
MAX_EMPTY_PAGES = 3

# ==================== PARAMÈTRES APEC ====================

# Filtre de date : 101852 = offres des 30 derniers jours
APEC_DATE_FILTER = "101852"

# URL de base APEC
APEC_BASE_URL = "https://www.apec.fr"
APEC_SEARCH_URL = f"{APEC_BASE_URL}/candidat/recherche-emploi.html/emploi"

# ==================== OPTIONS SELENIUM ====================

CHROME_OPTIONS = [
    "--headless=new",
    "--window-size=1920,1080",
    "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "--disable-logging",
    "--log-level=3",
    "--disable-images",
    "--disable-gpu",
    "--no-sandbox",
]

# ==================== PARAMÈTRES D'EXPORT ====================

# Dossier de sortie pour les fichiers JSON
OUTPUT_DIR = "output"

# Encodage des fichiers
FILE_ENCODING = "utf-8"

# Indentation JSON
JSON_INDENT = 2

# ==================== CHAMPS NLP À EXTRAIRE ====================

NLP_FIELDS = [
    "company_name",
    "reference_apec",
    "reference_company",
    "contract_type",
    "location",
    "publication_date",
    "update_date",
    "is_recent",
    "salary",
    "start_date",
    "experience_required",
    "experience_years",
    "job_title",
    "status",
    "travel_zone",
    "sector",
    "languages",
    "soft_skills",
    "hard_skills",
    "job_description",
    "required_profile",
    "company_description",
    "recruiter",
]

# ==================== TYPES DE CONTRAT ====================

CONTRACT_TYPES = ['CDI', 'CDD', 'Stage', 'Alternance', 'Freelance', 'Intérim']

# ==================== LANGUES SUPPORTÉES ====================

LANGUAGES = ['Anglais', 'Français', 'Espagnol', 'Allemand', 'Italien', 'Chinois', 'Arabe', 'Portugais']

# ==================== MESSAGES ====================

MESSAGES = {
    "start": "🚀 SCRAPER APEC OPTIMISÉ - ARCHITECTURE MODULAIRE",
    "collecting_urls": "📋 ÉTAPE 1: Collecte des URLs (< 30 jours)",
    "scraping": "⚡ ÉTAPE 2: Scraping parallèle",
    "nlp": "🧠 ÉTAPE 3: Application du NLP",
    "completed": "✅ Scraping terminé avec succès!",
}