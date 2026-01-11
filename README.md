# 🐝 RUCHE
Projet NLP & Text Mining – Master 2 SISE (2025–2026) 

### Application de cherche d'emplois

![Logo](streamlit/static/Logo3.png)

## Présentation du projet

**RUCHE** est une plateforme d’analyse du marché de l’emploi **Data Science & Intelligence Artificielle** en France.  
Elle combine **web scraping**, **NLP**, **machine learning**, **data warehousing** et **visualisation interactive** pour proposer :

-  **Une recherche sémantique intelligente** d’offres d’emploi  
-  **Une cartographie géographique interactive** du marché de l’emploi  
-  **Des analyses avancées** sur les salaires, les compétences et les tendances du marché
-  **Enregistrement de nouvelles offres** pour les utilisateurs de l'application

Le système repose sur une **architecture end-to-end**, depuis la collecte des données jusqu’à leur exploitation analytique au sein d’une application **Streamlit**.

## Architecture

```
RUCHE/
├── data/
│   ├── backup_job_market.duckdb
│   └── local.duckdb
│
├── scraping/
│   ├── francetravail/
│   ├── apec/
│   ├── jobteaser/
│   └── service_public/
│
├── mongodb/
│   ├── main_mongo.py
│   ├── reference_apec.py
│   ├── mongodb_load_jobteaser.py
│   └── mongodb_utils.py
│
├── etl/
│   ├── cleanX.py #Tout les "clean" fpnction de nettoyage de donnée
│   ├── config_etl.py
│   ├── etl_utils.py
│   ├── etl_vectorization.py
│   ├── tfidf_ml_data_filter.py
│   ├── geolocation_enrichment.py # API pour longétude et latitude 
│   └── etl_motherduck.py
│
├── streamlit_app/
│   ├── 1_home_page.py
│   ├── 2_cartographie.py
│   ├── 3_visualisation.py
│   ├── 4_add_offers.py
│   ├── 5_clustering.py
│   ├── 6_graphe_competences.py
│   ├── 7_llm.py
│   ├── 8_about.py
│   ├── app.py
│   ├── config.py
│   ├── static/ # Logo & images
│   ├── db/
│   └── analyse_competences/
│
├── docs/
│   ├── Rapport.md
│   ├── notice_france_travail_scraper.md
│   ├── notice_TFIDF_ML_filtre_data_nondata.md
│   └── notice_moteur_recherche_semantique.md
│
├── duck_to_mother.py
├── pyproject.toml
├── requirements.txt
├── test_connexion_duckdb.py
├── test_creation_duckdb.py
└── README.md

```

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements_mongodo_ftscraper.txt
```

2. **Configure `.env` file:**
```env
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/
FT_CLIENT_ID=your_ft_client_id
FT_CLIENT_SECRET=your_ft_client_secret
```

3. **Test MongoDB connection:**
```bash
python mongodb.mongodb_utils.py
```

## Usage

### France Travail Scraper
```bash
python scraper_francetravail.py
```

### Create Your Own Scraper
```python
from mongodb.mongodb_utils import get_collection, create_unique_index, bulk_upsert

# 1. Define your collection name
COLLECTION_NAME = "apec_raw"

# 2. Get collection
collection = get_collection(COLLECTION_NAME)
create_unique_index(collection, "id")

# 3. Scrape your data
offers = scrape_apec_data()  # Your scraping logic

# 4. Convert to list of dicts
documents = [offer.to_dict() for offer in offers]

# 5. Upsert to MongoDB
bulk_upsert(collection, documents)
```

## MongoDB Utilities Reference

### Connection Functions
- `get_mongo_client()` - Get MongoDB client
- `get_collection(name)` - Get specific collection
- `create_unique_index(collection, field)` - Create unique index

### Data Operations
- `bulk_upsert(collection, docs)` - Upsert documents (update or insert)
- `bulk_insert(collection, docs)` - Insert new documents only
- `count_documents(collection, filter)` - Count documents
- `get_latest_scraped(collection, limit)` - Get recent documents

### Collection Management
- `list_collections()` - List all collections
- `get_collection_stats(collection)` - Get collection statistics
- `drop_collection(name, confirm=True)` - Delete collection

## Best Practices

1. **Always use upsert** - Prevents duplicates
2. **Create unique index on 'id'** - Required for efficient upserts
3. **Use same DB_NAME** - All scrapers share `RUCHE_datalake`
4. **Different collections** - Each source gets its own collection
5. **Add scraped_at timestamp** - Track when data was collected

## Example: APEC Scraper Template
```python
from mongodb.mongodb_utils import get_collection, bulk_upsert
from dataclasses import dataclass, asdict

COLLECTION_NAME = "apec_raw"

@dataclass
class APECOffer:
    id: str
    title: str
    company: str
    # ... your fields

collection = get_collection(COLLECTION_NAME)
offers = scrape_apec()
docs = [offer.to_dict() for offer in offers]
bulk_upsert(collection, docs)
```
