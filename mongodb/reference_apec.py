"""
Script d'import MongoDB utilisant reference_apec comme identifiant unique
Collection: RUCHE_datalake > apec_raw
"""

import json
import os
from typing import List, Dict, Any
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv
import certifi
from collections import Counter

# Charger les variables d'environnement
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "RUCHE_datalake"


def load_and_prepare_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Charger le JSON et préparer les données avec reference_apec comme id
    
    Returns:
        Liste de documents prêts pour MongoDB
    """
    print(f"\n📂 Chargement: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"✗ Fichier introuvable!")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convertir en liste
        if isinstance(data, dict):
            data = [data]
        
        print(f"✓ {len(data)} documents chargés")
        
        # Préparer les données
        print(f"\n🔧 Préparation des données...")
        
        prepared_docs = []
        missing_ref = 0
        duplicate_refs = {}
        seen_refs = set()
        
        for i, doc in enumerate(data):
            if not isinstance(doc, dict):
                continue
            
            # Vérifier reference_apec
            if 'reference_apec' not in doc:
                missing_ref += 1
                print(f"   ⚠️  Document {i} sans 'reference_apec', ignoré")
                continue
            
            ref_apec = doc['reference_apec']
            
            # Ajouter/remplacer le champ 'id' avec reference_apec
            doc['id'] = ref_apec
            
            # Détecter les doublons dans le fichier
            if ref_apec in seen_refs:
                duplicate_refs[ref_apec] = duplicate_refs.get(ref_apec, 1) + 1
            else:
                seen_refs.add(ref_apec)
                prepared_docs.append(doc)
        
        # Statistiques
        print(f"\n📊 STATISTIQUES :")
        print(f"   - Documents valides : {len(prepared_docs)}")
        print(f"   - Documents sans reference_apec : {missing_ref}")
        print(f"   - References uniques : {len(seen_refs)}")
        
        if duplicate_refs:
            print(f"\n⚠️  DOUBLONS DANS LE FICHIER :")
            print(f"   - {len(duplicate_refs)} references en double (ignorées)")
            for ref, count in list(duplicate_refs.items())[:5]:
                print(f"     • {ref} : {count + 1} fois")
        
        return prepared_docs
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return []


def import_to_mongodb(
    json_file_path: str,
    mode: str = "insert",  # "insert" ou "upsert"
    collection_name: str = "apec_raw"
):
    """
    Importer les données dans MongoDB
    
    Args:
        json_file_path: Chemin du fichier JSON
        mode: "insert" (ignorer doublons) ou "upsert" (mettre à jour)
        collection_name: Nom de la collection
    """
    
    print("=" * 80)
    print("IMPORT MONGODB - UTILISANT reference_apec COMME ID")
    print(f"Collection: {DB_NAME}.{collection_name}")
    print(f"Mode: {mode.upper()}")
    print("=" * 80)
    
    # ÉTAPE 1 : Charger et préparer les données
    documents = load_and_prepare_data(json_file_path)
    
    if not documents:
        print("\n✗ Aucune donnée à importer!")
        return
    
    # ÉTAPE 2 : Connexion MongoDB
    print(f"\n🔌 Connexion à MongoDB Atlas...")
    try:
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
        client.admin.command('ping')
        print(f"✓ Connecté à {DB_NAME}")
    except Exception as e:
        print(f"✗ Erreur de connexion: {e}")
        return
    
    # ÉTAPE 3 : Accéder à la collection
    try:
        db = client[DB_NAME]
        collection = db[collection_name]
        print(f"✓ Collection: {collection_name}")
    except Exception as e:
        print(f"✗ Erreur d'accès: {e}")
        client.close()
        return
    
    # ÉTAPE 4 : Créer l'index unique sur 'id'
    print(f"\n🔑 Création de l'index unique sur 'id' (reference_apec)...")
    try:
        collection.create_index([("id", 1)], unique=True)
        collection.create_index([("reference_apec", 1)], unique=True)
        print("✓ Index créés")
    except Exception as e:
        print(f"⚠️  Warning: {e}")
    
    # ÉTAPE 5 : Vérifier les doublons avec la base
    print(f"\n🔍 Vérification des doublons dans la base...")
    try:
        doc_ids = [doc['id'] for doc in documents]
        existing = collection.count_documents({"id": {"$in": doc_ids}})
        new = len(doc_ids) - existing
        
        print(f"   - À importer : {len(doc_ids)}")
        print(f"   - Déjà en base : {existing}")
        print(f"   - Nouveaux : {new}")
    except Exception as e:
        print(f"   ⚠️  Erreur de vérification: {e}")
    
    # ÉTAPE 6 : Import
    print(f"\n📥 Import de {len(documents)} documents...")
    
    try:
        if mode == "upsert":
            print("Mode UPSERT: mise à jour ou insertion")
            
            operations = [
                UpdateOne(
                    {"id": doc["id"]},
                    {"$set": doc},
                    upsert=True
                )
                for doc in documents
            ]
            
            result = collection.bulk_write(operations, ordered=False)
            
            print(f"✓ {result.upserted_count} documents insérés")
            print(f"✓ {result.modified_count} documents mis à jour")
            
        else:  # insert
            print("Mode INSERT: ignorer les doublons")
            
            inserted = 0
            duplicates = 0
            errors = 0
            
            for doc in documents:
                try:
                    collection.insert_one(doc)
                    inserted += 1
                except Exception as e:
                    if "duplicate key error" in str(e).lower():
                        duplicates += 1
                    else:
                        errors += 1
                        if errors <= 3:
                            print(f"   ⚠️  Erreur: {str(e)[:100]}")
            
            print(f"✓ {inserted} nouveaux documents")
            if duplicates > 0:
                print(f"ℹ️  {duplicates} doublons ignorés")
            if errors > 0:
                print(f"✗ {errors} erreurs")
    
    except Exception as e:
        print(f"✗ Erreur d'import: {e}")
    
    # ÉTAPE 7 : Statistiques finales
    print(f"\n📊 Statistiques de la collection '{collection_name}':")
    try:
        total = collection.count_documents({})
        stats = db.command("collStats", collection_name)
        
        print(f"   - Total documents : {total}")
        print(f"   - Taille : {stats.get('size', 0) / 1024:.2f} KB")
        print(f"   - Taille moyenne : {stats.get('avgObjSize', 0)} bytes")
        print(f"   - Index : {stats.get('nindexes', 0)}")
        
        # Exemples de documents
        print(f"\n📄 Exemples de documents (3 premiers) :")
        for doc in collection.find().limit(3):
            print(f"   - ID: {doc.get('id')} | Ref APEC: {doc.get('reference_apec')}")
        
    except Exception as e:
        print(f"   ⚠️  Erreur stats: {e}")
    
    client.close()
    print("\n✓ Connexion fermée")
    
    print("\n" + "=" * 80)
    print("✓ IMPORT TERMINÉ!")
    print("=" * 80)


def clean_collection(collection_name: str = "apec_raw", confirm: bool = False):
    """
    Supprimer tous les documents de la collection
    
    Args:
        collection_name: Nom de la collection
        confirm: Doit être True pour confirmer la suppression
    """
    if not confirm:
        print("⚠️  Suppression annulée (confirm=False)")
        return False
    
    print(f"\n🗑️  Suppression de la collection '{collection_name}'...")
    
    try:
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
        db = client[DB_NAME]
        
        count_before = db[collection_name].count_documents({})
        db[collection_name].drop()
        
        print(f"✓ Collection supprimée ({count_before} documents)")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return False


def main():
    """Fonction principale"""
    
    # CONFIGURATION
    JSON_FILE = r"C:\Users\gopit\OneDrive\Documents\MASTER2SISE\Projet_NLP\RUCHE\output\data_cleaned.json"
    
    # =========================================================================
    # OPTION 1 : Nettoyer la collection existante (ATTENTION: supprime tout!)
    # =========================================================================
    
    print("\n⚠️  Voulez-vous SUPPRIMER la collection existante ?")
    print("   (Cela supprimera tous les documents actuels)")
    choice = input("   Taper 'OUI' pour confirmer: ")
    
    if choice == "OUI":
        clean_collection("apec_raw", confirm=True)
        print("\n")
    
    # =========================================================================
    # OPTION 2 : Import des données
    # =========================================================================
    
    import_to_mongodb(
        json_file_path=JSON_FILE,
        mode="insert",  # Ou "upsert" pour mettre à jour les existants
        collection_name="apec_raw"
    )


if __name__ == "__main__":
    main()