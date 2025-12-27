"""
Script pour corriger les IDs dans le fichier JSON
Remplace 'id' par 'reference_apec' comme identifiant unique
"""

import json
import os
from typing import List, Dict, Any

def fix_json_ids(input_file: str, output_file: str = None):
    """
    Corriger les IDs dans le fichier JSON en utilisant reference_apec
    
    Args:
        input_file: Fichier JSON source
        output_file: Fichier JSON de sortie (si None, écrase l'original)
    """
    
    print("=" * 80)
    print("CORRECTION DES IDs DANS LE FICHIER JSON")
    print("=" * 80)
    
    # Charger le fichier
    print(f"\n📂 Chargement: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"✗ Fichier introuvable!")
        return
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convertir en liste si nécessaire
        if isinstance(data, dict):
            data = [data]
        
        print(f"✓ {len(data)} documents chargés")
        
        # Analyser et corriger
        print(f"\n🔧 Correction des IDs...")
        
        fixed_count = 0
        missing_ref_count = 0
        duplicate_refs = {}
        seen_refs = set()
        
        for i, doc in enumerate(data):
            if not isinstance(doc, dict):
                continue
            
            # Vérifier si reference_apec existe
            if 'reference_apec' in doc:
                ref_apec = doc['reference_apec']
                
                # Remplacer 'id' par 'reference_apec'
                doc['id'] = ref_apec
                fixed_count += 1
                
                # Détecter les doublons
                if ref_apec in seen_refs:
                    duplicate_refs[ref_apec] = duplicate_refs.get(ref_apec, 1) + 1
                else:
                    seen_refs.add(ref_apec)
            else:
                missing_ref_count += 1
                print(f"   ⚠️  Document {i} sans 'reference_apec'")
        
        # Statistiques
        print(f"\n📊 RÉSULTATS :")
        print(f"   - Documents corrigés : {fixed_count}")
        print(f"   - Documents sans reference_apec : {missing_ref_count}")
        print(f"   - References uniques : {len(seen_refs)}")
        
        if duplicate_refs:
            print(f"\n⚠️  DOUBLONS DÉTECTÉS :")
            print(f"   - {len(duplicate_refs)} reference_apec en double")
            print(f"   Exemples (5 premiers) :")
            for ref, count in list(duplicate_refs.items())[:5]:
                print(f"     • {ref} : {count + 1} fois")
        
        # Sauvegarder
        if output_file is None:
            output_file = input_file
        
        print(f"\n💾 Sauvegarde dans: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Fichier sauvegardé!")
        
        print("\n" + "=" * 80)
        print("✓ CORRECTION TERMINÉE!")
        print("=" * 80)
        
        return {
            "total": len(data),
            "fixed": fixed_count,
            "missing": missing_ref_count,
            "unique_refs": len(seen_refs),
            "duplicates": len(duplicate_refs)
        }
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return None


def main():
    """Fonction principale"""
    
    # CONFIGURATION
    INPUT_FILE = r"C:\Users\gopit\OneDrive\Documents\MASTER2SISE\Projet_NLP\RUCHE\output\data_cleaned.json"
    
    # Option 1: Écraser le fichier original
    # fix_json_ids(INPUT_FILE)
    
    # Option 2: Créer un nouveau fichier
    OUTPUT_FILE = r"C:\Users\gopit\OneDrive\Documents\MASTER2SISE\Projet_NLP\RUCHE\output\data_cleaned_fixed.json"
    fix_json_ids(INPUT_FILE, OUTPUT_FILE)


if __name__ == "__main__":
    main()