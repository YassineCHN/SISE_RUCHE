"""
Module d'extraction NLP des offres APEC.
Extraction rapide sans Selenium, juste du parsing de texte.
"""

import re
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from config import CONTRACT_TYPES, LANGUAGES


def extract_field_after_keyword(content: str, keyword: str, multiline: bool = False) -> Optional[str]:
    """Extrait le contenu qui suit un mot-clé après un \\n."""
    if multiline:
        pattern = rf'{re.escape(keyword)}\s*\n(.*?)(?:\n\n|\n[A-Z][a-zéèêà]+\s*\n|$)'
    else:
        pattern = rf'{re.escape(keyword)}\s*\n([^\n]+)'
    
    match = re.search(pattern, content, re.DOTALL)
    if match:
        result = match.group(1).strip()
        if result and result != keyword:
            return result
    return None


def extract_company_name(content: str) -> Optional[str]:
    """Extrait le nom de l'entreprise."""
    match = re.search(r'Ref\. Apec[^\n]+\n([A-Z][^\n]+?)\n', content)
    if match:
        company = match.group(1).strip()
        if not company.isdigit() and len(company) > 2:
            return company
    return None


def extract_reference_apec(content: str) -> Optional[str]:
    """Extrait la référence APEC."""
    match = re.search(r'Ref\. Apec\s*:\s*(\w+)', content)
    return match.group(1).strip() if match else None


def extract_reference_company(content: str) -> Optional[str]:
    """Extrait la référence société."""
    match = re.search(r'Ref\. Société\s*:\s*(\w+)', content)
    return match.group(1).strip() if match else None


def extract_contract_type(content: str) -> Optional[str]:
    """Extrait le type de contrat."""
    for contract in CONTRACT_TYPES:
        if re.search(rf'\n{contract}\s*\n', content):
            return contract
    return None


def extract_location(content: str) -> Optional[str]:
    """Extrait la localisation (format: 'Ville - Code' ou 'Ville')."""
    for contract in CONTRACT_TYPES:
        location = extract_field_after_keyword(content, contract)
        if location and not location.startswith('Publiée') and not location.startswith('Actualisée'):
            if re.search(r'-\s*\d{2,3}', location) or (location[0].isupper() and not any(char.isdigit() for char in location)):
                return location
    return None


def extract_publication_date(content: str) -> Optional[str]:
    """Extrait la date de publication (format: JJ/MM/AAAA)."""
    match = re.search(r'Publiée le\s+(\d{2}/\d{2}/\d{4})', content)
    return match.group(1) if match else None


def extract_update_date(content: str) -> Optional[str]:
    """Extrait la date de mise à jour."""
    match = re.search(r'Actualisée le\s+(\d{2}/\d{2}/\d{4})', content)
    return match.group(1) if match else None


def is_recent_offer(publication_date: Optional[str], days: int = 30) -> bool:
    """Vérifie si l'offre a moins de X jours."""
    if not publication_date:
        return False
    try:
        date_obj = datetime.strptime(publication_date, "%d/%m/%Y")
        cutoff_date = datetime.now() - timedelta(days=days)
        return date_obj >= cutoff_date
    except:
        return False


def extract_experience_years(content: str) -> Optional[int]:
    """Extrait le nombre d'années d'expérience minimum."""
    exp = extract_field_after_keyword(content, 'Expérience')
    if exp:
        match = re.search(r'(\d+)\s*ans?', exp)
        if match:
            return int(match.group(1))
    return None


def extract_languages(content: str) -> List[str]:
    """Extrait les langues requises."""
    lang_section = extract_field_after_keyword(content, 'Langues', multiline=True)
    if not lang_section:
        return []
    if "Aucune langue attendue" in lang_section:
        return ["Aucune"]
    
    languages = []
    for lang in LANGUAGES:
        if lang in lang_section:
            languages.append(lang)
    return languages


def extract_soft_skills(content: str) -> List[str]:
    """Extrait les savoir-être."""
    section = extract_field_after_keyword(content, 'Savoir-être', multiline=True)
    if not section:
        return []
    
    skills = []
    for line in section.split('\n'):
        line = line.strip()
        if line and len(line) > 3 and line not in ['Voir plus', 'Savoir-faire', 'Entreprise']:
            skills.append(line)
    return skills


def extract_hard_skills(content: str) -> List[str]:
    """Extrait les savoir-faire (compétences techniques)."""
    section = extract_field_after_keyword(content, 'Savoir-faire', multiline=True)
    if not section:
        return []
    
    skills = []
    for line in section.split('\n'):
        line = line.strip()
        if not line or line in ['Voir plus', 'Entreprise', 'Autres offres', 'Débutant', 'Confirmé', 'Expert']:
            continue
        if 'Minimum' in line or 'ans d' in line:
            continue
        skills.append(line)
    return skills


def extract_job_description(content: str) -> Optional[str]:
    """Extrait le descriptif du poste."""
    desc = extract_field_after_keyword(content, 'Descriptif du poste', multiline=True)
    if desc and 'Profil recherché' in desc:
        desc = desc.split('Profil recherché')[0].strip()
    return desc


def extract_required_profile(content: str) -> Optional[str]:
    """Extrait le profil recherché."""
    profile = extract_field_after_keyword(content, 'Profil recherché', multiline=True)
    if profile and 'Compétences attendues' in profile:
        profile = profile.split('Compétences attendues')[0].strip()
    return profile


def extract_company_description(content: str) -> Optional[str]:
    """Extrait la description de l'entreprise."""
    desc = extract_field_after_keyword(content, 'Entreprise', multiline=True)
    if desc:
        for keyword in ['Autres offres', 'Personne en charge du recrutement']:
            if keyword in desc:
                desc = desc.split(keyword)[0].strip()
    return desc


def extract_recruiter(content: str) -> Optional[str]:
    """Extrait le nom du recruteur."""
    return extract_field_after_keyword(content, 'Personne en charge du recrutement')


def apply_nlp_extraction(content: str) -> Dict:
    """
    Applique toutes les extractions NLP sur le contenu brut.
    
    Args:
        content: Texte brut de l'offre
        
    Returns:
        Dictionnaire avec toutes les informations extraites
    """
    return {
        "company_name": extract_company_name(content),
        "reference_apec": extract_reference_apec(content),
        "reference_company": extract_reference_company(content),
        "contract_type": extract_contract_type(content),
        "location": extract_location(content),
        "publication_date": extract_publication_date(content),
        "update_date": extract_update_date(content),
        "is_recent": is_recent_offer(extract_publication_date(content), days=30),
        "salary": extract_field_after_keyword(content, 'Salaire'),
        "start_date": extract_field_after_keyword(content, 'Prise de poste'),
        "experience_required": extract_field_after_keyword(content, 'Expérience'),
        "experience_years": extract_experience_years(content),
        "job_title": extract_field_after_keyword(content, 'Métier'),
        "status": extract_field_after_keyword(content, 'Statut du poste'),
        "travel_zone": extract_field_after_keyword(content, 'Zone de déplacement'),
        "sector": extract_field_after_keyword(content, "Secteur d'activité du poste"),
        "languages": extract_languages(content),
        "soft_skills": extract_soft_skills(content),
        "hard_skills": extract_hard_skills(content),
        "job_description": extract_job_description(content),
        "required_profile": extract_required_profile(content),
        "company_description": extract_company_description(content),
        "recruiter": extract_recruiter(content),
    }


def enrich_offers(raw_offers: List[Dict]) -> List[Dict]:
    """
    Applique le NLP sur une liste d'offres brutes.
    
    Args:
        raw_offers: Liste d'offres avec description_raw
        
    Returns:
        Liste d'offres enrichies avec extraction NLP
    """
    enriched = []
    
    for i, offer in enumerate(raw_offers, 1):
        content = offer.get('description_raw', '')
        nlp_data = apply_nlp_extraction(content)
        
        enriched_offer = {
            "source": "apec",
            **offer,
            "content_length": len(content),
            **nlp_data,
            "extraction_success": True
        }
        
        enriched.append(enriched_offer)
        
        # Affichage progression
        if i % 50 == 0 or i == len(raw_offers):
            pct = i / len(raw_offers) * 100
            print(f"   Traité: {i}/{len(raw_offers)} ({pct:.1f}%)", end='\r')
    
    print(f"\n   ✅ NLP appliqué sur {len(enriched)} offres\n")
    return enriched