"""
Module for cleaning localisation data.
The goal is to extract and normalize city names from various location formats.
"""

import re
import pandas as pd


def normalize_city_name(city: str) -> str:
    """
    Normalize the name of a city: first letter uppercase, rest lowercase
    Handles special cases (d', -sur-, etc.)
    Examples:
        'PARIS' -> 'Paris'
        'saint-denis' -> 'Saint-Denis'
        "aix-en-provence" -> "Aix-en-Provence"
    """
    if not city or city == 'UNKNOWN':
        return city
    
    # Mots à garder en minuscule (articles, prépositions)
    lowercase_words = {'le', 'la', 'les', 'de', 'du', 'des', 'en', 'sur', 'sous', 'au', 'aux'}
    
    # Split sur les espaces et tirets
    parts = []
    for segment in city.split():
        # Gérer les tirets
        if '-' in segment:
            subsegments = segment.split('-')
            normalized_subsegments = []
            for i, sub in enumerate(subsegments):
                sub_lower = sub.lower()
                # Gérer "d'xxx" spécialement
                if sub_lower.startswith("d'") and len(sub_lower) > 2:
                    normalized_subsegments.append("d'" + sub_lower[2:].capitalize())
                # Premier segment ou pas dans la liste des mots en minuscule
                elif i == 0 or sub_lower not in lowercase_words:
                    normalized_subsegments.append(sub_lower.capitalize())
                else:
                    normalized_subsegments.append(sub_lower)
            parts.append('-'.join(normalized_subsegments))
        else:
            word_lower = segment.lower()
            # Gérer "d'" spécialement
            if word_lower.startswith("d'") and len(word_lower) > 2:
                parts.append("d'" + word_lower[2:].capitalize())
            elif word_lower in lowercase_words and len(parts) > 0:
                parts.append(word_lower)
            else:
                parts.append(word_lower.capitalize())
    
    return ' '.join(parts)


def extract_city_from_location(location: str) -> str:
    """
    Extrait le nom de ville à partir de différents formats de localisation
    
    Formats gérés:
    - "Cannes (France)" -> "Cannes"
    - "Terr. Boieldieu, 92800 Puteaux (France)" -> "Puteaux"
    - "91120 Palaiseau (France)" -> "Palaiseau"
    - "Crolles, 38140, FR (France)" -> "Crolles"
    - "78140 Vélizy-Villacoublay (France)" -> "Vélizy-Villacoublay"
    - "12 Rue Pierre-Félix Delarue, 72100 Le Mans (France)" -> "Le Mans"
    - "Paris 01 - 75" -> "Paris"
    - "Saint-Didier-au-Mont-d'Or - 69" -> "Saint-Didier-au-Mont-d'Or"
    - "75 - Paris 1er Arrondissement" -> "Paris"
    - "75 - Paris" -> "Paris"
    - "Montpellier" -> "Montpellier"
    - "Saint-Maurice, 12 rue du Val d'Osne" -> "Saint-Maurice"
    """
    
    if not location or pd.isna(location):
        return 'UNKNOWN'
    
    location = str(location).strip()
    
    if not location:
        return 'UNKNOWN'
    
    # Supprimer "(France)", "(FR)", etc. à la fin
    location = re.sub(r'\s*\([^)]*\)\s*$', '', location)
    
    # Pattern 1: "75 - Paris 1er Arrondissement" ou "75 - Paris"
    # Extraire ce qui suit le code département
    match = re.match(r'^\d{2}[AB]?\s*-\s*(.+?)(?:\s+\d+(?:er|ème|e)?\s+Arrondissement)?$', location, re.IGNORECASE)
    if match:
        city = match.group(1).strip()
        # Enlever un éventuel " -" à la fin
        city = re.sub(r'\s*-\s*$', '', city)
        return normalize_city_name(city)
    
    # Pattern 2: "Saint-Didier-au-Mont-d'Or - 69" (ville avant le code)
    # ou "Paris 01 - 75" (ville + arrondissement avant le code)
    match = re.match(r'^([A-Za-zÀ-ÿ\s\-\']+?)\s+\d*\s*-\s*\d{2}[AB]?\s*$', location)
    if match:
        city = match.group(1).strip()
        # Enlever un éventuel numéro d'arrondissement à la fin
        city = re.sub(r'\s+\d+$', '', city)
        return normalize_city_name(city)
    
    # Pattern 3: Code postal suivi de la ville: "92800 Puteaux" ou "78140 Vélizy-Villacoublay"
    match = re.search(r'\b\d{5}\s+([A-Za-zÀ-ÿ\s\-\']+?)(?:\s*,|$)', location)
    if match:
        city = match.group(1).strip()
        return normalize_city_name(city)
    
    # Pattern 4: Adresse complexe avec numéro de rue et code postal
    # "12 Rue Pierre-Félix Delarue, 72100 Le Mans"
    match = re.search(r',\s*\d{5}\s+([A-Za-zÀ-ÿ\s\-\']+?)(?:\s*,|$)', location)
    if match:
        city = match.group(1).strip()
        return normalize_city_name(city)
    
    # Pattern 5: "Ville, code_postal" (ville en premier)
    # "Crolles, 38140, FR" ou "Saint-Maurice, 12 rue du Val d'Osne"
    match = re.match(r'^([A-Za-zÀ-ÿ\s\-\']+?)\s*,', location)
    if match:
        city = match.group(1).strip()
        # Vérifier que ce n'est pas un numéro de rue
        if not re.match(r'^\d+', city):
            return normalize_city_name(city)
    
    # Pattern 6: Juste le nom de ville (sans virgule, sans code)
    # "Montpellier"
    match = re.match(r'^([A-Za-zÀ-ÿ\s\-\']+)$', location)
    if match:
        city = match.group(0).strip()
        return normalize_city_name(city)
    
    # Si aucun pattern ne correspond, essayer de prendre le dernier mot significatif
    words = location.split(',')[-1].strip()
    words = re.sub(r'\d{5}', '', words).strip()  # Enlever code postal
    words = re.sub(r'\b\d{2}[AB]?\b', '', words).strip()  # Enlever code département
    
    if words and not words.isdigit():
        return normalize_city_name(words)
    
    return 'UNKNOWN'


def clean_location_column(df: pd.DataFrame, location_col: str = 'location') -> pd.DataFrame:
    """
    Nettoie la colonne location d'un DataFrame
    
    Args:
        df: DataFrame contenant la colonne à nettoyer
        location_col: nom de la colonne contenant les localisations
    
    Returns:
        DataFrame avec colonne nettoyée
    """
    df = df.copy()
    
    print(f" Cleaning {location_col} column...")
    print(f"   Before: {df[location_col].nunique()} unique values")
    
    # Appliquer l'extraction
    df[location_col] = df[location_col].apply(extract_city_from_location)
    
    print(f"   After:  {df[location_col].nunique()} unique values")
    print(f"   UNKNOWN entries: {(df[location_col] == 'UNKNOWN').sum()}")
    
    return df


# ============================================================================
# Tests unitaires
# ============================================================================

def test_extraction():
    """Tests des différents formats"""
    test_cases = [
        ("Cannes (France)", "Cannes"),
        ("Terr. Boieldieu, 92800 Puteaux (France)", "Puteaux"),
        ("91120 Palaiseau (France)", "Palaiseau"),
        ("Crolles, 38140, FR (France)", "Crolles"),
        ("78140 Vélizy-Villacoublay (France)", "Vélizy-Villacoublay"),
        ("12 Rue Pierre-Félix Delarue, 72100 Le Mans (France)", "Le Mans"),
        ("Paris 01 - 75", "Paris"),
        ("Saint-Didier-au-Mont-d'Or - 69", "Saint-Didier-au-Mont-d'Or"),
        ("75 - Paris 1er Arrondissement", "Paris"),
        ("75 - Paris", "Paris"),
        ("Montpellier", "Montpellier"),
        ("Saint-Maurice, 12 rue du Val d'Osne", "Saint-Maurice"),
        ("", "UNKNOWN"),
        (None, "UNKNOWN"),
        ("LYON", "Lyon"),
        ("aix-en-provence", "Aix-en-Provence"),
        ("saint-jean-de-luz", "Saint-Jean-de-Luz"),
    ]
    
    print("\n" + "="*70)
    print("TESTS D'EXTRACTION DE VILLE")
    print("="*70)
    
    passed = 0
    failed = 0
    
    for input_loc, expected in test_cases:
        result = extract_city_from_location(input_loc)
        status = "✅" if result == expected else "❌"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
            
        print(f"{status} '{input_loc}' -> '{result}' (expected: '{expected}')")
    
    print("="*70)
    print(f"Résultats: {passed} réussis, {failed} échoués")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    # Exécuter les tests
    success = test_extraction()
    
    if success:
        print("✅ Tous les tests sont passés!")
    else:
        print("❌ Certains tests ont échoué")