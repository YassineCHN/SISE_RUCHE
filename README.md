# RUCHE Data Lake - Job Scrapers

## Architecture
```
RUCHE_datalake (MongoDB Database)
├── francetravail_raw    # France Travail job offers
├── apec_raw             # APEC job offers
├── jobteaser_raw        # JobTeaser job offers
└── ...                  # Other sources
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
python mongodb_utils.py
```

## Usage

### France Travail Scraper
```bash
python scraper_francetravail.py
```

### Create Your Own Scraper
```python
from mongodb_utils import get_collection, create_unique_index, bulk_upsert

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
from mongodb_utils import get_collection, bulk_upsert
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