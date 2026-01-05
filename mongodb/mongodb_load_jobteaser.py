from pathlib import Path
import json
from mongodb.mongodb_utils import get_collection, create_unique_index, bulk_upsert

PROJECT_ROOT = Path(__file__).resolve().parents[1]

JSON_FILE = (
    PROJECT_ROOT / "scrapers/jobteaser/output/jobteaser_enriched_20260104_170404.json"
)
COLLECTION_NAME = "jobteaser_raw"


def load_enriched_jobteaser(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    valid_docs = []
    missing_id = 0

    for doc in data:
        if "id" not in doc:
            missing_id += 1
            continue
        valid_docs.append(doc)

    print(f"[JOBTEASER] Documents total : {len(data)}")
    print(f"[JOBTEASER] Documents valides : {len(valid_docs)}")
    if missing_id > 0:
        print(f"[WARNING] {missing_id} documents sans 'id' ignorés")

    return valid_docs


def main():
    documents = load_enriched_jobteaser(JSON_FILE)
    if not documents:
        print("❌ Aucun document à importer")
        return

    collection = get_collection(COLLECTION_NAME)
    if collection is None:
        return

    # Index unique sur id
    create_unique_index(collection, "id")

    print("\n📥 Import MongoDB (UPSERT)")
    total = bulk_upsert(collection, documents, id_field="id")
    print(f"✅ {total} documents insérés / mis à jour")


if __name__ == "__main__":
    main()
