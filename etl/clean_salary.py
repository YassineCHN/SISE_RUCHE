"""
Module for cleaning and standardizing salary data.
The goal is to extract and normalize salary information from various formats
and categorize them for easy filtering in Streamlit.

Author: Ruche's team
Date: 2026-01-10
"""

import re
import pandas as pd
from typing import Tuple, Optional


# ============================================================================
# SALARY CATEGORIES CONFIGURATION
# ============================================================================

SALARY_RANGES = [
    ("< 25k€", 0, 25000),
    ("25k€ - 30k€", 25000, 30000),
    ("30k€ - 35k€", 30000, 35000),
    ("35k€ - 40k€", 35000, 40000),
    ("40k€ - 45k€", 40000, 45000),
    ("45k€ - 50k€", 45000, 50000),
    ("50k€ - 60k€", 50000, 60000),
    ("60k€ - 70k€", 60000, 70000),
    ("70k€ - 80k€", 70000, 80000),
    ("80k€ - 100k€", 80000, 100000),
    ("> 100k€", 100000, float('inf'))
]


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def extract_salary_info(salary_str: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Extrait les informations de salaire à partir de différents formats
    
    Formats gérés:
    - "45 - 60 k€ brut annuel" -> (45000, 60000, 52500)
    - "A partir de 58 k€ brut annuel" -> (58000, None, 58000)
    - "Mensuel de 2900.0 Euros à 3000.0 Euros sur 12.0 mois" -> (34800, 36000, 35400)
    - "Annuel de 48000.0 Euros à 50000.0 Euros" -> (48000, 50000, 49000)
    - "Horaire de 17.0 Euros à 30.0 Euros" -> (30940, 54600, 42770)
    - "1500 EUR - 1700 EUR / mois" -> (18000, 20400, 19200)
    - "1400 EUR / mois" -> (16800, 16800, 16800)
    - "36000-42000" -> (36000, 42000, 39000)
    - "45000" -> (45000, 45000, 45000)
    - "A négocier" -> (None, None, None)
    - "Selon expérience" -> (None, None, None)
    
    Args:
        salary_str: Chaîne de caractères contenant le salaire
    
    Returns:
        Tuple (min_annual, max_annual, avg_annual) en euros
        Retourne (None, None, None) si le salaire n'est pas spécifié ou non parsable
    """
    if not salary_str or str(salary_str).strip() == '':
        return None, None, None
    
    salary_str = str(salary_str).strip()
    
    # Cas de non-renseignement
    non_specified_keywords = [
        'non renseigné', 'information non renseignée', 
        'selon expérience', 'selon profil', 'a négocier',
        'à négocier', 'variable', 'non communiqué'
    ]
    
    if any(keyword in salary_str.lower() for keyword in non_specified_keywords):
        return None, None, None
    
    # Pattern 1: APEC range - "45 - 60 k€ brut annuel"
    match = re.search(r'(\d+)\s*-\s*(\d+)\s*k€\s*brut\s*annuel', salary_str)
    if match:
        min_sal = float(match.group(1)) * 1000
        max_sal = float(match.group(2)) * 1000
        avg_sal = (min_sal + max_sal) / 2
        return min_sal, max_sal, avg_sal
    
    # Pattern 2: APEC from - "A partir de 45 k€ brut annuel"
    match = re.search(r'A partir de (\d+)\s*k€\s*brut\s*annuel', salary_str)
    if match:
        min_sal = float(match.group(1)) * 1000
        return min_sal, None, min_sal
    
    # Pattern 3: France Travail mensuel - "Mensuel de 2900.0 Euros à 3000.0 Euros"
    match = re.search(r'Mensuel de ([\d.]+)\s*Euros(?:\s*à\s*([\d.]+)\s*Euros)?', salary_str)
    if match:
        min_sal = float(match.group(1)) * 12
        max_sal = float(match.group(2)) * 12 if match.group(2) else min_sal
        avg_sal = (min_sal + max_sal) / 2
        return min_sal, max_sal, avg_sal
    
    # Pattern 4: France Travail annuel - "Annuel de 48000.0 Euros à 50000.0 Euros"
    match = re.search(r'Annuel de ([\d.]+)\s*Euros(?:\s*à\s*([\d.]+)\s*Euros)?', salary_str)
    if match:
        min_sal = float(match.group(1))
        max_sal = float(match.group(2)) if match.group(2) else min_sal
        avg_sal = (min_sal + max_sal) / 2
        return min_sal, max_sal, avg_sal
    
    # Pattern 5: France Travail horaire - "Horaire de 17.0 Euros à 30.0 Euros"
    # Estimation: 35h/semaine * 52 semaines = 1820 heures/an
    match = re.search(r'Horaire de ([\d.]+)\s*Euros(?:\s*à\s*([\d.]+)\s*Euros)?', salary_str)
    if match:
        min_sal = float(match.group(1)) * 1820
        max_sal = float(match.group(2)) * 1820 if match.group(2) else min_sal
        avg_sal = (min_sal + max_sal) / 2
        return min_sal, max_sal, avg_sal
    
    # Pattern 6: JobTeaser mensuel range - "1500 EUR - 1700 EUR / mois"
    match = re.search(r'(\d+)\s*EUR\s*-\s*(\d+)\s*EUR\s*/\s*mois', salary_str)
    if match:
        min_sal = float(match.group(1)) * 12
        max_sal = float(match.group(2)) * 12
        avg_sal = (min_sal + max_sal) / 2
        return min_sal, max_sal, avg_sal
    
    # Pattern 7: JobTeaser mensuel single - "1400 EUR / mois"
    match = re.search(r'(\d+)\s*EUR\s*/\s*mois', salary_str)
    if match:
        sal = float(match.group(1)) * 12
        return sal, sal, sal
    
    # Pattern 8: Simple range - "36000-42000" ou "2800-3500"
    match = re.search(r'^(\d+)-(\d+)$', salary_str)
    if match:
        min_val = float(match.group(1))
        max_val = float(match.group(2))
        
        # Si < 10000, probablement mensuel
        if max_val < 10000:
            min_sal = min_val * 12
            max_sal = max_val * 12
        else:
            min_sal = min_val
            max_sal = max_val
        
        avg_sal = (min_sal + max_sal) / 2
        return min_sal, max_sal, avg_sal
    
    # Pattern 9: Simple single - "45000" (un seul nombre de 5-6 chiffres)
    match = re.search(r'^(\d{5,6})$', salary_str)
    if match:
        sal = float(match.group(1))
        return sal, sal, sal
    
    # Si aucun pattern ne correspond
    return None, None, None


def get_salary_category(avg_salary: Optional[float]) -> str:
    """
    Catégorise un salaire moyen annuel en tranches
    
    Args:
        avg_salary: Salaire moyen annuel en euros
    
    Returns:
        Catégorie de salaire (ex: "45k€ - 50k€")
    
    Examples:
        >>> get_salary_category(47000)
        '45k€ - 50k€'
        >>> get_salary_category(None)
        'Non spécifié'
        >>> get_salary_category(120000)
        '> 100k€'
    """
    if avg_salary is None:
        return "Non spécifié"
    
    for label, min_val, max_val in SALARY_RANGES:
        if min_val <= avg_salary < max_val:
            return label
    
    return "A négocier"


def standardize_salary_column(df: pd.DataFrame, salary_col: str = 'salary') -> pd.DataFrame:
    """
    Standardise la colonne salary d'un DataFrame
    Ajoute UNIQUEMENT la colonne: salaire
    
    Args:
        df: DataFrame contenant la colonne à standardiser
        salary_col: nom de la colonne contenant les salaires
    
    Returns:
        DataFrame avec la nouvelle colonne salaire
    
    Example:
        >>> df = pd.DataFrame({'salary': ['45 - 60 k€ brut annuel', 'Non renseigné']})
        >>> df = standardize_salary_column(df)
        >>> df.columns
        Index(['salary', 'salaire'], dtype='object')
    """
    df = df.copy()
    
    print(f"STEP X.X: Categorizing {salary_col} column...")
    print(f"   Total records: {len(df)}")
    
    # Appliquer l'extraction sur toutes les lignes
    salary_data = df[salary_col].apply(extract_salary_info)
    
    # Extraire seulement le salaire moyen pour la catégorisation
    avg_salaries = salary_data.apply(lambda x: x[2])  # x[2] = avg_annual
    
    # Catégoriser les salaires
    df['salaire'] = avg_salaries.apply(get_salary_category)
    
    # Statistiques
    parsed = avg_salaries.notna().sum()
    not_specified = (df['salaire'] == 'Non spécifié').sum()
    
    print(f"   ✓ Categorized successfully: {parsed}/{len(df)} ({parsed/len(df)*100:.1f}%)")
    print(f"   ✓ Not specified: {not_specified} ({not_specified/len(df)*100:.1f}%)")
    
    # Distribution par catégorie
    print(f"   ✓ Distribution by salary category:")
    cat_dist = df['salaire'].value_counts()
    for cat, count in cat_dist.items():
        print(f"      - {cat}: {count} ({count/len(df)*100:.1f}%)")
    
    return df


# ============================================================================
# TESTS UNITAIRES
# ============================================================================

def test_salary_extraction():
    """Tests des différents formats de salaires"""
    test_cases = [
        # (input, expected_min, expected_max, expected_avg)
        ("45 - 60 k€ brut annuel", 45000, 60000, 52500),
        ("A partir de 58 k€ brut annuel", 58000, None, 58000),
        ("Mensuel de 2900.0 Euros à 3000.0 Euros sur 12.0 mois", 34800, 36000, 35400),
        ("Mensuel de 40000.0 Euros sur 12.0 mois", 480000, 480000, 480000),
        ("Annuel de 48000.0 Euros à 50000.0 Euros sur 12.0 mois", 48000, 50000, 49000),
        ("Annuel de 50000.0 Euros à 70000.0 Euros sur 12.0 mois", 50000, 70000, 60000),
        ("Horaire de 17.0 Euros à 30.0 Euros sur 12.0 mois", 30940, 54600, 42770),
        ("1500 EUR - 1700 EUR / mois", 18000, 20400, 19200),
        ("1400 EUR / mois", 16800, 16800, 16800),
        ("36000-42000", 36000, 42000, 39000),
        ("3143-3403", 37716, 40836, 39276),
        ("45000", 45000, 45000, 45000),
        ("27798", 27798, 27798, 27798),
        ("Non renseigné", None, None, None),
        ("Selon expérience", None, None, None),
        ("A négocier", None, None, None),
        ("", None, None, None),
        (None, None, None, None),
    ]
    
    print("\n" + "="*80)
    print("TESTS D'EXTRACTION DE SALAIRE")
    print("="*80)
    
    passed = 0
    failed = 0
    
    for input_sal, expected_min, expected_max, expected_avg in test_cases:
        min_sal, max_sal, avg_sal = extract_salary_info(input_sal)
        
        # Comparer avec tolérance pour les floats
        min_ok = (min_sal == expected_min) if expected_min is None else (
            abs(min_sal - expected_min) < 1 if min_sal is not None else False
        )
        max_ok = (max_sal == expected_max) if expected_max is None else (
            abs(max_sal - expected_max) < 1 if max_sal is not None else False
        )
        avg_ok = (avg_sal == expected_avg) if expected_avg is None else (
            abs(avg_sal - expected_avg) < 1 if avg_sal is not None else False
        )
        
        status = "✅" if (min_ok and max_ok and avg_ok) else "❌"
        
        if min_ok and max_ok and avg_ok:
            passed += 1
        else:
            failed += 1
        
        # Formatage de l'affichage
        input_display = f"'{input_sal}'" if input_sal else "None"
        result_display = f"({min_sal}, {max_sal}, {avg_sal})"
        expected_display = f"({expected_min}, {expected_max}, {expected_avg})"
        
        print(f"{status} {input_display}")
        print(f"   Result:   {result_display}")
        if not (min_ok and max_ok and avg_ok):
            print(f"   Expected: {expected_display}")
    
    print("="*80)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*80 + "\n")
    
    return failed == 0


def test_salary_categorization():
    """Tests de catégorisation des salaires"""
    test_cases = [
        (20000, "< 25k€"),
        (27000, "25k€ - 30k€"),
        (35000, "35k€ - 40k€"),
        (47000, "45k€ - 50k€"),
        (55000, "50k€ - 60k€"),
        (75000, "70k€ - 80k€"),
        (120000, "> 100k€"),
        (None, "Non spécifié"),
    ]
    
    print("\n" + "="*80)
    print("TESTS DE CATÉGORISATION DE SALAIRE")
    print("="*80)
    
    passed = 0
    failed = 0
    
    for salary, expected_cat in test_cases:
        result = get_salary_category(salary)
        status = "✅" if result == expected_cat else "❌"
        
        if result == expected_cat:
            passed += 1
        else:
            failed += 1
        
        salary_display = f"{salary:,}" if salary else "None"
        print(f"{status} {salary_display} -> '{result}' (expected: '{expected_cat}')")
    
    print("="*80)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*80 + "\n")
    
    return failed == 0


# ============================================================================
# MAIN (pour tests)
# ============================================================================

if __name__ == "__main__":
    # Exécuter les tests
    print("="*80)
    print("MODULE clean_salary.py - UNIT TESTS")
    print("="*80)
    
    test1_success = test_salary_extraction()
    test2_success = test_salary_categorization()
    
    if test1_success and test2_success:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    # Test avec un DataFrame d'exemple
    print("\n" + "="*80)
    print("DATAFRAME STANDARDIZATION TEST")
    print("="*80)
    
    sample_data = pd.DataFrame({
        'salary': [
            '45 - 60 k€ brut annuel',
            'Mensuel de 2900.0 Euros à 3000.0 Euros sur 12.0 mois',
            'Non renseigné',
            '36000-42000',
            'A partir de 58 k€ brut annuel', 
            '1400 EUR / mois',
            'Selon expérience',
            '1500 EUR - 1700 EUR / mois',
            '45000'

        ]
    })
    
    print("\nBefore:")
    print(sample_data)
    
    result_df = standardize_salary_column(sample_data)
    
    print("\nAfter:")
    print(result_df[['salary', 'salaire']])