"""
Script pour importer des données JSON APEC dans MongoDB Atlas
Collection: RUCHE_datalake > apec_raw
Version corrigée utilisant reference_apec comme identifiant unique
"""

import json
import os
from typing import List, Dict, Any
from pymongo import MongoClient
from dotenv import load_dotenv
import certifi
from collections import Counter

# Charger les variables d'environnement
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "RUCHE_datalake"


def analyze_json_structure(file_path: str) -> Dict[str, Any]:
    """
    Analyser la structure du fichier JSON et détecter les problèmes
    
    Returns:
        Dictionnaire avec statistiques d'analyse
    """
    print(f"\n🔍 ANALYSE DU FICHIER JSON")
    print("=" * 80)
    
    if not os.path.exists(file_path):
        return {"error": f"Fichier introuvable: {file_path}"}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convertir en liste si nécessaire
        if isinstance(data, dict):
            data = [data]
        
        # Analyser les champs
        refs = []
        ids = []
        missing_ref = 0
        missing_id = 0
        
        for i, doc in enumerate(data):
            if not isinstance(doc, dict):
                continue
            
            # Vérifier reference_apec
            if 'reference_apec' in doc:
                refs.append(doc['reference_apec'])
            else:
                missing_ref += 1
            
            # Vérifier id
            if 'id' in doc:
                ids.append(doc['id'])
            else:
                missing_id += 1
        
        # Compter les doublons
        ref_counts = Counter(refs)
        id_counts = Counter(ids)
        
        ref_duplicates = {k: v for k, v in ref_counts.items() if v > 1}
        id_duplicates = {k: v for k, v in id_counts.items() if v > 1}
        
        stats = {
            "total_documents": len(data),
            "with_reference_apec": len(refs),
            "with_id": len(ids),
            "missing_reference_apec": missing_ref,
            "missing_id": missing_id,
            "unique_references": len(set(refs)),
            "unique_ids": len(set(ids)),
            "ref_duplicates": len(ref_duplicates),
            "id_duplicates": len(id_duplicates)
        }
        
        # Affichage
        print(f"\n📊 STRUCTURE DU FICHIER :")
        print(f"   - Total de documents : {stats['total_documents']}")
        print(f"   - Documents avec 'reference_apec' : {stats['with_reference_apec']}")
        print(f"   - Documents avec 'id' : {stats['with_id']}")
        print(f"   - References APEC uniques : {stats['unique_references']}")
        print(f"   - IDs uniques : {stats['unique_ids']}")
        
        if stats['missing_reference_apec'] > 0:
            print(f"\n⚠️  {stats['missing_reference_apec']} documents SANS 'reference_apec'")
        
        if stats['ref_duplicates'] > 0:
            print(f"\n⚠️  DOUBLONS 'reference_apec' DÉTECTÉS :")
            print(f"   {stats['ref_duplicates']} références en double")
            for ref, count in list(ref_duplicates.items())[:5]:
                print(f"   • {ref} : {count} fois")
        
        if stats['id_duplicates'] > 0:
            print(f"\n⚠️  DOUBLONS 'id' DÉTECTÉS :")
            print(f"   {stats['id_duplicates']} IDs en double")
            for doc_id, count in list(id_duplicates.items())[:5]:
                print(f"   • {doc_id} : {count} fois")
        
        print("=" * 80)
        
        return stats
        
    except Exception as e:
        return {"error": str(e)}


def load_and_prepare_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Charger le fichier JSON et préparer les documents
    Copie reference_apec vers id pour chaque document
    
    Returns:
        Liste de documents préparés
    """
    print(f"\n📂 CHARGEMENT DU FICHIER")
    print("=" * 80)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convertir en liste
        if isinstance(data, dict):
            data = [data]
        
        print(f"✓ {len(data)} documents chargés")
        
        # Préparer les documents
        print(f"\n🔧 PRÉPARATION DES DOCUMENTS")
        print("   Copie de 'reference_apec' vers 'id'...")
        
        prepared = []
        missing_ref = 0
        duplicates_found = {}
        seen_refs = set()
        
        for i, doc in enumerate(data):
            if not isinstance(doc, dict):
                continue
            
            # Vérifier reference_apec
            if 'reference_apec' not in doc:
                missing_ref += 1
                print(f"   ⚠️  Document {i} sans 'reference_apec', ignoré")
                continue
            
            ref = doc['reference_apec']
            
            # Détecter doublons dans le fichier
            if ref in seen_refs:
                if ref not in duplicates_found:
                    duplicates_found[ref] = 1
                duplicates_found[ref] += 1
                continue  # Ignorer le doublon
            
            # Copier reference_apec vers id
            doc['id'] = ref
            seen_refs.add(ref)
            prepared.append(doc)
        
        # Statistiques
        print(f"\n✓ Documents préparés : {len(prepared)}")
        if missing_ref > 0:
            print(f"⚠️  Documents ignorés (sans reference_apec) : {missing_ref}")
        if duplicates_found:
            print(f"⚠️  Doublons ignorés : {len(duplicates_found)}")
            for ref, count in list(duplicates_found.items())[:5]:
                print(f"   • {ref} : {count} fois")
        
        print("=" * 80)
        
        return prepared
        
    except Exception as e:
        print(f"✗ ERREUR: {e}")
        return []


def import_to_apec_raw(
    json_file_path: str,
    mode: str = "insert",  # "insert" ou "upsert"
    analyze_first: bool = True
):
    """
    Importer des données JSON dans la collection apec_raw
    
    Args:
        json_file_path: Chemin vers le fichier JSON
        mode: "insert" (ignorer doublons) ou "upsert" (mettre à jour)
        analyze_first: Analyser le fichier avant import
    """
    
    print("\n")
    print("=" * 80)
    print("IMPORT DE DONNÉES APEC DANS MONGODB ATLAS")
    print(f"Collection: {DB_NAME}.apec_raw")
    print(f"Mode: {mode.upper()}")
    print("=" * 80)
    
    # ÉTAPE 0 : Analyse (optionnelle)
    if analyze_first:
        stats = analyze_json_structure(json_file_path)
        if "error" in stats:
            print(f"\n✗ Erreur d'analyse: {stats['error']}")
            return
    
    # ÉTAPE 1 : Charger et préparer
    documents = load_and_prepare_json(json_file_path)
    
    if not documents:
        print("\n✗ Aucune donnée à importer!")
        return
    
    # ÉTAPE 2 : Connexion MongoDB
    print(f"\n🔌 CONNEXION À MONGODB")
    print("=" * 80)
    
    try:
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
        client.admin.command('ping')
        print(f"✓ Connecté à MongoDB Atlas")
        print(f"  Database: {DB_NAME}")
    except Exception as e:
        print(f"✗ Erreur de connexion: {e}")
        print("  Vérifiez MONGO_URI dans votre fichier .env")
        return
    
    # ÉTAPE 3 : Accès à la collection
    try:
        db = client[DB_NAME]
        collection = db["apec_raw"]
        print(f"✓ Collection: apec_raw")
    except Exception as e:
        print(f"✗ Erreur d'accès: {e}")
        client.close()
        return
    
    print("=" * 80)
    
    # ÉTAPE 4 : Création des index
    print(f"\n🔑 CRÉATION DES INDEX")
    print("=" * 80)
    
    try:
        collection.create_index([("id", 1)], unique=True)
        print("✓ Index unique sur 'id'")
        
        collection.create_index([("reference_apec", 1)], unique=True)
        print("✓ Index unique sur 'reference_apec'")
    except Exception as e:
        print(f"⚠️  Warning: {e}")
    
    print("=" * 80)
    
    # ÉTAPE 5 : Vérification des doublons en base
    print(f"\n🔍 VÉRIFICATION DES DOUBLONS")
    print("=" * 80)
    
    try:
        doc_ids = [doc['id'] for doc in documents]
        existing_docs = list(collection.find(
            {"id": {"$in": doc_ids}},
            {"id": 1, "_id": 0}
        ))
        existing_ids = [doc['id'] for doc in existing_docs]
        new_ids = [id for id in doc_ids if id not in existing_ids]
        
        print(f"   - Documents à importer : {len(doc_ids)}")
        print(f"   - Déjà en base : {len(existing_ids)}")
        print(f"   - Nouveaux : {len(new_ids)}")
        
        if len(existing_ids) > 0 and len(existing_ids) <= 5:
            print(f"\n   IDs déjà présents :")
            for eid in existing_ids:
                print(f"   • {eid}")
    
    except Exception as e:
        print(f"   ⚠️  Erreur de vérification: {e}")
    
    print("=" * 80)
    
    # ÉTAPE 6 : Import
    print(f"\n📥 IMPORT DES DOCUMENTS")
    print("=" * 80)
    
    try:
        if mode == "upsert":
            from pymongo import UpdateOne
            
            print(f"Mode UPSERT: Mise à jour ou insertion")
            
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
            print(f"Mode INSERT: Ignorer les doublons")
            
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
            
            print(f"✓ {inserted} nouveaux documents insérés")
            if duplicates > 0:
                print(f"ℹ️  {duplicates} doublons ignorés")
            if errors > 0:
                print(f"✗ {errors} erreurs")
    
    except Exception as e:
        print(f"✗ Erreur d'import: {e}")
    
    print("=" * 80)
    
    # ÉTAPE 7 : Statistiques finales
    print(f"\n📊 STATISTIQUES FINALES")
    print("=" * 80)
    
    try:
        total = collection.count_documents({})
        stats = db.command("collStats", "apec_raw")
        
        print(f"   - Total de documents : {total}")
        print(f"   - Taille de la collection : {stats.get('size', 0) / 1024:.2f} KB")
        print(f"   - Taille moyenne par document : {stats.get('avgObjSize', 0)} bytes")
        print(f"   - Nombre d'index : {stats.get('nindexes', 0)}")
        
        # Exemples
        print(f"\n📄 Exemples de documents (3 premiers) :")
        for doc in collection.find().limit(3):
            print(f"   • ID: {doc.get('id')[:20]}... | Ref: {doc.get('reference_apec')}")
    
    except Exception as e:
        print(f"   ⚠️  Erreur stats: {e}")
    
    print("=" * 80)
    
    # Fermeture
    client.close()
    print(f"\n✓ Connexion fermée")
    
    print("\n" + "=" * 80)
    print("✓ IMPORT TERMINÉ!")
    print("=" * 80)


def clean_collection(confirm_text: str = None):
    """
    Supprimer la collection apec_raw
    
    Args:
        confirm_text: Doit être "OUI" pour confirmer
    """
    if confirm_text != "OUI":
        print("⚠️  Suppression annulée")
        return
    
    print("\n🗑️  SUPPRESSION DE LA COLLECTION")
    print("=" * 80)
    
    try:
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
        db = client[DB_NAME]
        
        count = db["apec_raw"].count_documents({})
        db["apec_raw"].drop()
        
        print(f"✓ Collection 'apec_raw' supprimée ({count} documents)")
        print("=" * 80)
        
        client.close()
        
    except Exception as e:
        print(f"✗ Erreur: {e}")


def main():
    """
    Fonction principale
    """
    
    # CONFIGURATION
    JSON_FILE_PATH = r"C:\Users\gopit\OneDrive\Documents\MASTER2SISE\Projet_NLP\RUCHE\output\data_cleaned.json"
    
    # Demander confirmation pour nettoyer
    print("\n" + "=" * 80)
    print("NETTOYAGE DE LA COLLECTION (OPTIONNEL)")
    print("=" * 80)
    print("\n⚠️  Voulez-vous SUPPRIMER tous les documents de la collection 'apec_raw' ?")
    print("   (Utile si vous avez des données incorrectes à remplacer)")
    choice = input("\n   Tapez 'OUI' pour confirmer la suppression, ou Enter pour ignorer: ")
    
    if choice == "OUI":
        clean_collection("OUI")
    
    # Lancer l'import
    import_to_apec_raw(
        json_file_path=JSON_FILE_PATH,
        mode="insert",  # Changer en "upsert" pour mettre à jour les existants
        analyze_first=True
    )


if __name__ == "__main__":
    main()