"""
Module for cleaning localisation data.
The goal is to extract and normalize city names from various location formats.
"""

import re
import html
import pandas as pd

def normalize_city_name(city: str) -> str:
    """Normalise la casse (ex: 'SAINT-DENIS' -> 'Saint-Denis') et les abréviations."""
    if not city or city.upper() == 'UNKNOWN': return 'UNKNOWN'
    
    # Correction des abréviations courantes avant normalisation
    city = re.sub(r'\bs/\b', 'sur', city, flags=re.IGNORECASE)
    if re.search(r'\bNeuilly\b', city, re.IGNORECASE): return "Neuilly-sur-Seine"

    lowercase_words = {'le', 'la', 'les', 'de', 'du', 'des', 'en', 'sur', 'sous', 'au', 'aux'}
    
    def transform_word(word, index):
        word_low = word.lower()
        if word_low.startswith("d'") and len(word_low) > 2:
            return f"d'{word_low[2:].capitalize()}"
        if index > 0 and word_low in lowercase_words:
            return word_low
        return word_low.capitalize()

    city = re.sub(r'^[^A-Za-zÀ-ÿ]+|[^A-Za-zÀ-ÿ]+$', '', city)
    parts = []
    for i, space_part in enumerate(city.split()):
        if '-' in space_part:
            parts.append('-'.join([transform_word(sub, i + j) for j, sub in enumerate(space_part.split('-'))]))
        else:
            parts.append(transform_word(space_part, i))
    return ' '.join(parts)

def extract_city_from_location(location: str) -> str:
    """Moteur généraliste basé sur l'exclusion des types de voies."""
    if not location or pd.isna(location) or str(location).upper() == 'UNKNOWN': 
        return 'UNKNOWN'

    # 1. Nettoyage initial
    loc = html.unescape(str(location))
    loc = loc.replace('&#xa;', ' ').replace('\n', ' ').replace('\r', ' ')
    loc = re.sub(r'\s+', ' ', loc).strip()
    loc = re.sub(r'\s*\((?:FRANCE|FR|france|fr|France)\)$', '', loc, flags=re.IGNORECASE)

    # 2. Priorité Métropoles (Scan global)
    for major in ['Paris', 'Lyon', 'Marseille', 'Grenoble', 'Bordeaux', 'Toulouse', 'Lille', 'Nantes']:
        if re.search(rf'\b{major}\b', loc, re.IGNORECASE):
            return major

    # 3. Filtrage par segments (Séparateurs: virgule, tiret long, slash)
    segments = re.split(r',| - | / ', loc)
    street_keywords = r'\b(Rue|Av\.|Avenue|Bd|Boulevard|Place|Pl\.|Route|Chemin|Allée|Square|Impasse|Quai|Cours|Résidence|Bâtiment|Immeuble|Avenue|St)\b'
    
    potential_candidates = []
    for seg in segments:
        seg = seg.strip()
        # On ignore si c'est une rue ou purement numérique
        if not re.search(street_keywords, seg, re.IGNORECASE) and not re.match(r'^\d+$', seg):
            # Nettoyage des codes postaux (2 à 5 chiffres)
            clean_seg = re.sub(r'\b\d{2,5}\b', '', seg).strip()
            if len(clean_seg) > 2:
                potential_candidates.append(clean_seg)

    # 4. Décision sur le candidat
    if potential_candidates:
        # On privilégie le dernier candidat trouvé (souvent la ville en fin d'adresse)
        # Sauf si le premier est un nom de service (Sgami/Dipn), on traite via post-process
        candidate = potential_candidates[-1]
        if re.search(r'\b(Sgami|Dipn|Drpj|Ddsp)\b', candidate, re.IGNORECASE):
            m = re.search(r'\b(Nord|Sud|Est|Ouest)\b', loc, re.IGNORECASE)
            return m.group(0).capitalize() if m else candidate.split()[0].upper()
        return normalize_city_name(candidate)

    # 5. Fallback Regex CP + Ville
    p_cp = re.search(r'(?:\b\d{3,5}\b)\s*[-]?\s*([A-Za-zÀ-ÿ\s\-\']+)|([A-Za-zÀ-ÿ\s\-\']+?)\s*[-]?\s*(?:\b\d{3,5}\b)', loc)
    if p_cp:
        res = p_cp.group(1) if p_cp.group(1) else p_cp.group(2)
        return normalize_city_name(res.strip())

    return 'UNKNOWN'

# ============================================================================
# TESTS GÉNÉRAUX (Couvrant tous les scénarios précédents)
# ============================================================================

def run_general_tests():
    test_cases = [
        # Adresses et Voies
        ("47 Av. de la Grande Armée, 75016 Paris", "Paris"),
        ("12 Rue Pierre-Félix Delarue, 72100 Le Mans", "Le Mans"),
        ("Saint-Maurice, 12 rue du Val d'Osne", "Saint-Maurice"),
        ("135 Avenue Charles de Gaulle, Neuilly-sur-S", "Neuilly-sur-Seine"),
        
        # Administratif et Métropoles
        ("Paris 15eme", "Paris"),
        ("Paris Cedex 08", "Paris"),
        ("Marseille 10", "Marseille"),
        ("Grenoble - Sophia Antipolis - le Chesnay-Ro", "Grenoble"),
        ("Sgami Nord - Direction de L'immobilier", "UNKNOWN"),
        ("Dipn - Service Local de Police Judiciaire", "UNKNOWN"),
        
        # Formats Codes Postaux et DOM-TOM
        ("972 - Lamentin", "Lamentin"),
        ("Le Lamentin - 972", "Le Lamentin"),
        ("Bordeaux - 33", "Bordeaux"),
        ("Lyon - 69", "Lyon"),
        
        # Nettoyage et Cas Spéciaux
        ("/ Avenue des Champs Elysees&#xa; Paris&#xa;", "Paris"),
        ("Place du Marche St Honore&#xa; Paris&#xa;", "Paris"),
        ("aix-en-provence", "Aix-en-Provence"),
        ("Paris-Drouot", "Paris"),
        ("UNKNOWN", "UNKNOWN")
    ]
    
    print(f"{'INPUT':<45} | {'RESULT':<18} | STATUS")
    print("-" * 80)
    passed = 0
    for inp, exp in test_cases:
        res = extract_city_from_location(inp)
        status = "✅" if res == exp else "❌"
        if res == exp: passed += 1
        print(f"{str(inp)[:43]:<45} | {res:<18} | {status}")
        if res != exp: print(f"   ⚠️ Attendu: {exp}")
    
    print("-" * 80)
    print(f"SCORE GÉNÉRAL: {passed}/{len(test_cases)} ({(passed/len(test_cases))*100:.1f}%)")

if __name__ == "__main__":
    run_general_tests()