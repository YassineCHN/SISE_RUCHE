import json
import re

def clean_text(text):
    """
    Nettoie le texte en:
    - Supprimant les \n
    - Remplaçant les doubles espaces par un seul espace
    """
    if text is None:
        return None
    
    # Supprimer les \n et \t
    text = text.replace('\n', ' ').replace('\t', ' ')
    
    # Remplacer les doubles espaces (ou plus) par un seul espace
    text = re.sub(r'\s+', ' ', text)
    
    # Supprimer les espaces en début et fin
    text = text.strip()
    
    return text

def clean_job_offer(offer):
    """
    Nettoie une offre d'emploi en:
    - Supprimant les champs: description_raw, content_length, extraction_success
    - Nettoyant les champs: company_description, required_profile, job_description
    """
    # Supprimer les champs non désirés
    fields_to_remove = ['description_raw', 'content_length', 'extraction_success']
    for field in fields_to_remove:
        offer.pop(field, None)
    
    # Nettoyer les champs textuels
    fields_to_clean = ['company_description', 'required_profile', 'job_description']
    for field in fields_to_clean:
        if field in offer and offer[field] is not None:
            offer[field] = clean_text(offer[field])
    
    return offer

def process_job_offers(input_file, output_file):
    """
    Traite le fichier JSON d'offres d'emploi
    """
    # Lire le fichier JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Nettoyer chaque offre
    if isinstance(data, list):
        cleaned_data = [clean_job_offer(offer.copy()) for offer in data]
    else:
        cleaned_data = clean_job_offer(data.copy())
    
    # Écrire le résultat
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f" Traitement terminé!")
    print(f" Fichier d'entrée: {input_file}")
    print(f" Fichier de sortie: {output_file}")
    
    if isinstance(cleaned_data, list):
        print(f" Nombre d'offres traitées: {len(cleaned_data)}")

if __name__ == "__main__":
    # Exemple d'utilisation
    input_file = r"RUCHE\output\raw_scraping.json"  # À adapter
    output_file = r"RUCHE\output\data_cleaned.json"
    
    process_job_offers(input_file, output_file)