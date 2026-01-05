"""
MongoDB Utilities - Reusable functions for MongoDB Atlas
Shared across all scrapers (France Travail, APEC, JobTeaser, etc.)
"""

import os
import certifi
from typing import Optional, List, Dict, Any
from pymongo import MongoClient, UpdateOne, ASCENDING
from dotenv import load_dotenv

# Load environment
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# Shared database name
DB_NAME = "RUCHE_datalake"


# =============================================================================
# MONGODB CONNECTION
# =============================================================================

def get_mongo_client() -> Optional[MongoClient]:
    """
    Create and return a MongoDB client connected to Atlas
    
    Returns:
        MongoClient instance or None if connection fails
    """
    try:
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
        # Test connection
        client.admin.command('ping')
        print(f"[DB] Connected to MongoDB Atlas (Database: {DB_NAME})")
        return client
    except Exception as e:
        print(f"[DB ERROR] Could not connect to MongoDB: {e}")
        return None


def get_collection(collection_name: str, client: Optional[MongoClient] = None):
    """
    Get a specific collection from the shared database
    
    Args:
        collection_name: Name of the collection (e.g., "francetravail_raw", "apec_raw")
        client: Optional existing MongoClient. If None, creates a new one.
    
    Returns:
        MongoDB collection object or None if connection fails
    """
    try:
        # Create client if not provided
        if client is None:
            client = get_mongo_client()
            if client is None:
                return None
        
        # Access database and collection
        db = client[DB_NAME]
        collection = db[collection_name]
        
        print(f"[DB] Using collection: {DB_NAME}.{collection_name}")
        return collection
        
    except Exception as e:
        print(f"[DB ERROR] Could not access collection '{collection_name}': {e}")
        return None


def create_unique_index(collection, field_name: str = "id") -> bool:
    """
    Create a unique index on a field to prevent duplicates and speed up queries
    
    Args:
        collection: MongoDB collection object
        field_name: Field to index (default: "id")
    
    Returns:
        True if successful, False otherwise
    """
    if collection is None:  # ← CORRECTION : Comparaison explicite
        print("[DB ERROR] Cannot create index on None collection")
        return False
    
    try:
        collection.create_index([(field_name, ASCENDING)], unique=True)
        print(f"[DB] Unique index created on '{field_name}' (if not exists)")
        return True
    except Exception as e:
        print(f"[DB WARNING] Index creation issue: {e}")
        return False


# =============================================================================
# BULK OPERATIONS
# =============================================================================

def bulk_upsert(collection, documents: List[Dict[str, Any]], id_field: str = "id") -> int:
    """
    Bulk upsert (update or insert) documents into MongoDB collection
    
    Strategy: If document with same id_field exists, UPDATE it. Otherwise, INSERT.
    
    Args:
        collection: MongoDB collection object
        documents: List of dictionaries to upsert
        id_field: Field to use as unique identifier (default: "id")
    
    Returns:
        Number of documents upserted/modified
    """
    if not documents or collection is None:  # ← Correct (comparaison explicite)
        return 0

    operations = []
    
    for doc in documents:
        # Ensure the id_field exists in the document
        if id_field not in doc:
            print(f"[DB WARNING] Document missing '{id_field}' field, skipping")
            continue
        
        # Prepare UpdateOne operation with upsert
        operations.append(
            UpdateOne(
                {id_field: doc[id_field]},  # Filter: find by id_field
                {"$set": doc},              # Action: set all fields
                upsert=True                 # Insert if not found
            )
        )
    
    if not operations:
        return 0
    
    try:
        result = collection.bulk_write(operations, ordered=False)
        total_ops = result.upserted_count + result.modified_count
        
        if result.upserted_count > 0:
            print(f"[DB] Inserted {result.upserted_count} new documents")
        if result.modified_count > 0:
            print(f"[DB] Updated {result.modified_count} existing documents")
        
        return total_ops
        
    except Exception as e:
        print(f"[DB WRITE ERROR] {e}")
        return 0


def bulk_insert(collection, documents: List[Dict[str, Any]], ignore_duplicates: bool = True) -> int:
    """
    Bulk insert documents (no update, only insert new ones)
    
    Args:
        collection: MongoDB collection object
        documents: List of dictionaries to insert
        ignore_duplicates: If True, skip duplicates silently. If False, raise error.
    
    Returns:
        Number of documents inserted
    """
    if not documents or collection is None:  # ← Correct (comparaison explicite)
        return 0
    
    try:
        result = collection.insert_many(documents, ordered=False)
        inserted = len(result.inserted_ids)
        print(f"[DB] Inserted {inserted} new documents")
        return inserted
        
    except Exception as e:
        if ignore_duplicates and "duplicate key error" in str(e).lower():
            print(f"[DB] Some duplicates skipped")
            return 0
        else:
            print(f"[DB INSERT ERROR] {e}")
            return 0


# =============================================================================
# QUERY HELPERS
# =============================================================================

def count_documents(collection, filter_query: Optional[Dict] = None) -> int:
    """
    Count documents in collection
    
    Args:
        collection: MongoDB collection object
        filter_query: Optional filter (e.g., {"type_contrat": "CDI"})
    
    Returns:
        Number of documents matching filter
    """
    if collection is None:  # ← CORRECTION : Comparaison explicite
        print("[DB ERROR] Cannot count on None collection")
        return 0
    
    try:
        count = collection.count_documents(filter_query or {})
        return count
    except Exception as e:
        print(f"[DB COUNT ERROR] {e}")
        return 0


def get_latest_scraped(collection, limit: int = 10) -> List[Dict]:
    """
    Get the most recently scraped documents
    
    Args:
        collection: MongoDB collection object
        limit: Number of documents to return
    
    Returns:
        List of documents sorted by scraped_at (descending)
    """
    if collection is None:  # ← CORRECTION : Comparaison explicite
        print("[DB ERROR] Cannot query None collection")
        return []
    
    try:
        cursor = collection.find().sort("scraped_at", -1).limit(limit)
        return list(cursor)
    except Exception as e:
        print(f"[DB QUERY ERROR] {e}")
        return []


def delete_old_documents(collection, days: int = 90) -> int:
    """
    Delete documents older than X days (based on scraped_at field)
    
    Args:
        collection: MongoDB collection object
        days: Number of days to keep (delete older)
    
    Returns:
        Number of documents deleted
    """
    if collection is None:  # ← CORRECTION : Comparaison explicite
        print("[DB ERROR] Cannot delete from None collection")
        return 0
    
    from datetime import datetime, timedelta
    
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        result = collection.delete_many({
            "scraped_at": {"$lt": cutoff_date.isoformat()}
        })
        deleted = result.deleted_count
        print(f"[DB] Deleted {deleted} documents older than {days} days")
        return deleted
    except Exception as e:
        print(f"[DB DELETE ERROR] {e}")
        return 0


# =============================================================================
# COLLECTION MANAGEMENT
# =============================================================================

def list_collections(client: Optional[MongoClient] = None) -> List[str]:
    """
    List all collections in the shared database
    
    Args:
        client: Optional existing MongoClient
    
    Returns:
        List of collection names
    """
    try:
        if client is None:
            client = get_mongo_client()
            if client is None:
                return []
        
        db = client[DB_NAME]
        collections = db.list_collection_names()
        
        print(f"[DB] Collections in {DB_NAME}:")
        for coll in collections:
            count = db[coll].count_documents({})
            print(f"  - {coll}: {count} documents")
        
        return collections
        
    except Exception as e:
        print(f"[DB ERROR] Could not list collections: {e}")
        return []


def drop_collection(collection_name: str, confirm: bool = False) -> bool:
    """
    Drop (delete) an entire collection
    
    WARNING: This deletes ALL documents in the collection permanently!
    
    Args:
        collection_name: Name of collection to drop
        confirm: Must be True to actually drop (safety check)
    
    Returns:
        True if successful, False otherwise
    """
    if not confirm:
        print(f"[DB WARNING] Drop operation requires confirm=True")
        return False
    
    try:
        client = get_mongo_client()
        if client is None:
            return False
        
        db = client[DB_NAME]
        db.drop_collection(collection_name)
        print(f"[DB] Collection '{collection_name}' dropped")
        return True
        
    except Exception as e:
        print(f"[DB ERROR] Could not drop collection: {e}")
        return False


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_collection_stats(collection) -> Dict[str, Any]:
    """
    Get statistics about a collection
    
    Returns:
        Dictionary with stats (size, count, indexes, etc.)
    """
    if collection is None:  # ← CORRECTION : Comparaison explicite
        print("[DB ERROR] Cannot get stats on None collection")
        return {}
    
    try:
        stats = collection.database.command("collStats", collection.name)
        return {
            "count": stats.get("count", 0),
            "size_bytes": stats.get("size", 0),
            "avg_doc_size": stats.get("avgObjSize", 0),
            "storage_size": stats.get("storageSize", 0),
            "indexes": stats.get("nindexes", 0),
        }
    except Exception as e:
        print(f"[DB STATS ERROR] {e}")
        return {}


def test_connection() -> bool:
    """
    Test MongoDB connection
    
    Returns:
        True if connection successful, False otherwise
    """
    client = get_mongo_client()
    if client is not None:  # ← CORRECTION : Comparaison explicite
        print("[DB] Connection test: SUCCESS")
        client.close()
        return True
    else:
        print("[DB] Connection test: FAILED")
        return False


# =============================================================================
# EXAMPLE USAGE (for testing this module)
# =============================================================================

if __name__ == "__main__":
    """Test the MongoDB utilities"""
    
    print("="*80)
    print("TESTING MONGODB UTILITIES")
    print("="*80)
    
    # Test 1: Connection
    print("\n[TEST 1] Testing connection...")
    test_connection()
    
    # Test 2: List collections
    print("\n[TEST 2] Listing collections...")
    list_collections()
    
    # Test 3: Create a test collection
    print("\n[TEST 3] Creating test collection...")
    test_collection = get_collection("test_collection")
    
    if test_collection is not None:  # ← CORRECTION PRINCIPALE
        # Create index
        create_unique_index(test_collection, "id")
        
        # Insert test documents
        test_docs = [
            {"id": "TEST001", "name": "Test Job 1", "salary": "40K"},
            {"id": "TEST002", "name": "Test Job 2", "salary": "50K"},
        ]
        
        print("\n[TEST 4] Bulk upsert test documents...")
        bulk_upsert(test_collection, test_docs)
        
        # Count
        print("\n[TEST 5] Counting documents...")
        count = count_documents(test_collection)
        print(f"Total documents: {count}")
        
        # Get stats
        print("\n[TEST 6] Collection stats...")
        stats = get_collection_stats(test_collection)
        print(f"Stats: {stats}")
        
        # Cleanup (optional)
        print("\n[TEST 7] Cleanup test collection...")
        response = input("Drop test collection? [y/N]: ")
        if response.lower() == 'y':
            drop_collection("test_collection", confirm=True)
    else:
        print("[ERROR] Could not create test collection")
    
    print("\n" + "="*80)
    print("TESTS COMPLETE")
    print("="*80)