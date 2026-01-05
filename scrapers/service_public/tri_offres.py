import json
import spacy

# Chargement du modèle NLP français
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    import os
    os.system("python -m spacy download fr_core_news_sm")
    nlp = spacy.load("fr_core_news_sm")

def contains_keywords(text, keywords):
    """
    Vérifie si le texte contient les mots-clés en utilisant la lemmatisation.
    """
    if not text:
        return False
    
    # Prétraitement NLP du texte (mise en minuscule et analyse)
    doc = nlp(text.lower())
    
    # On vérifie si le lemme ou le texte brut du mot est dans nos mots-clés
    for token in doc:
        # On vérifie le mot tel quel et sa version racine (lemme)
        if token.text in keywords or token.lemma_ in keywords:
            return True
    return False

def trier_offres(input_file, output_file):
    # Liste des mots-clés (on utilise les racines pour plus d'efficacité)
    keywords_titre = {"data", "ia", "donnée", "donnee"}
    keywords_desc = {"data", "ia", "donnée", "donnee"}
    
    try:
        # 1. Chargement du fichier source
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        offres_filtrees = []
        
        # 2. Parcours et filtrage
        print(f"🔍 Analyse de {len(data)} offres en cours...")
        
        for item in data:
            titre = item.get("titre", "")
            description = item.get("description", "")
            
            # Vérification des conditions
            match_titre = contains_keywords(titre, keywords_titre)
            match_desc = contains_keywords(description, keywords_desc)
            
            if match_titre or match_desc:
                offres_filtrees.append(item)
        
        # 3. Sauvegarde des résultats
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(offres_filtrees, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Tri terminé !")
        print(f"📊 {len(offres_filtrees)} offres conservées sur {len(data)}.")
        print(f"💾 Fichier créé : {output_file}")

    except FileNotFoundError:
        print(f" Erreur : Le fichier {input_file} est introuvable.")
    except Exception as e:
        print(f" Une erreur est survenue : {e}")

# Lancement du script
if __name__ == "__main__":
    trier_offres("offres_service_public.json", "offres_service_public_tri.json")