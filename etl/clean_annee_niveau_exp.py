"""
Experience Data Cleaning Module - For ETL Integration
======================================================
Simple functions to clean experience data in ETL pipeline

Author: Ruche's teams
Date: 2026-01-11
Version: 2.1 (ETL Compatible)
"""

import re
import pandas as pd
from typing import Optional, Union


def clean_nb_annees_experience(value: Union[int, float, str, None]) -> Optional[int]:
    """
    Convert months to years and handle outliers.
    
    Rules:
    - NULL/empty → None
    - Value > 12 → divide by 12 (months to years)
    - Result > 10 → cap at 10
    - Value ≤ 12 → keep as is (already in years)
    
    Examples:
        18 → 2, 24 → 2, 180 → 10, 5 → 5
    """
    if pd.isna(value) or value is None or value == '':
        return None
    
    exp = float(value)
    
    if exp > 12:
        exp_years = exp / 12
        return 10 if exp_years > 10 else int(round(exp_years))
    else:
        return int(round(exp))


def categorize_by_years(years: float) -> str:
    """
    Map years to experience level category.
    
    0 → Débutant
    1-2 → Junior
    3-5 → Confirmé
    6-9 → Senior
    10+ → Expert
    """
    if years == 0:
        return 'Débutant'
    elif years <= 2:
        return 'Junior'
    elif years <= 5:
        return 'Confirmé'
    elif years <= 9:
        return 'Senior'
    else:
        return 'Expert'


def extract_experience_level(text: str) -> str:
    """
    Extract experience level from text.
    
    Returns one of:
    - Débutant (0 ans)
    - Junior (1-2 ans)
    - Confirmé (3-5 ans)
    - Senior (6-9 ans)
    - Expert (10+ ans ou PhD)
    - À préciser avec l'entreprise (indéterminable)
    """
    if pd.isna(text) or text is None or text == '':
        return "À préciser avec l'entreprise"
    
    text_lower = str(text).lower()
    
    # Priority 1: Explicit keywords
    if any(kw in text_lower for kw in ['débutant', 'premiere experience', 'première expérience', 'stage', 'sans expérience']):
        return 'Débutant'
    
    if 'expert' in text_lower:
        years_match = re.search(r'(\d+)\s*(?:an|année)', text_lower)
        if years_match:
            years = int(years_match.group(1))
            if years >= 10:
                return 'Expert'
            elif years >= 5:
                return 'Senior'
        return 'Expert'
    
    if 'confirmé' in text_lower:
        return 'Confirmé'
    
    # Priority 2: PhD/Doctorat
    if any(kw in text_lower for kw in ['doctorat', 'phd', 'thèse', 'docteur']):
        return 'Expert'
    
    # Priority 3: Special cases
    if 'tous niveaux' in text_lower or 'tous les niveaux' in text_lower:
        return 'Junior'
    
    # Priority 4: Standalone numbers
    if re.match(r'^\d+$', text_lower.strip()):
        years = int(text_lower.strip())
        return categorize_by_years(years)
    
    # Priority 5: Experience keywords without numbers
    if any(kw in text_lower for kw in [
        'expérience avérée', 'expérience exigée', 'expérience similaire',
        'expérience professionnelle avérée', 'expérience technique',
        'expérience significative'
    ]):
        if not re.search(r'\d+', text_lower):
            return 'Confirmé'
    
    # Priority 6: Job-specific keywords without numbers
    if any(kw in text_lower for kw in [
        'expérience sur', 'expérience dans', 'gestion de projet',
        'management', 'pilotage', 'leadership', 'coordination'
    ]):
        if not re.search(r'\d+', text_lower) and 'bac' not in text_lower:
            return 'Confirmé'
    
    # Priority 7: Extract years from patterns
    # Pattern: "X An(s)", "X ans", "X années"
    years_match = re.search(r'(\d+)\s*(?:an\(s\)|ans?|année)', text_lower)
    if years_match:
        return categorize_by_years(int(years_match.group(1)))
    
    # Pattern: "X Mois"
    months_match = re.search(r'(\d+)\s*mois', text_lower)
    if months_match:
        months = int(months_match.group(1))
        return categorize_by_years(months / 12)
    
    # Pattern: "X à Y ans", "X-Y ans"
    range_match = re.search(r'(\d+)\s*(?:à|-)\s*(\d+)', text_lower)
    if range_match:
        min_years = int(range_match.group(1))
        max_years = int(range_match.group(2))
        avg_years = (min_years + max_years) / 2
        return categorize_by_years(avg_years)
    
    # Pattern: "minimum X ans", "plus de X ans"
    minimum_match = re.search(r'(?:minimum|plus de|minimum de)\s*(\d+)', text_lower)
    if minimum_match:
        return categorize_by_years(int(minimum_match.group(1)))
    
    # Priority 8: Special text cases
    if any(kw in text_lower for kw in ['aucune expérience', '0 an', 'non renseigné']):
        return 'Débutant'
    
    if 'expérience souhaitée' in text_lower or 'expérience bienvenue' in text_lower:
        return 'Junior'
    
    return "À préciser avec l'entreprise"


def clean_experience_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean experience columns in the DataFrame.
    
    Modifies:
    - nb_annees_experience: converts months to years, caps at 10
    - experience_required: replaces with categories (Débutant/Junior/Confirmé/Senior/Expert)
    
    Returns:
        DataFrame with cleaned experience columns
    """
    print("\nSTEP 6.5: Cleaning experience data...")
    print("-" * 60)
    
    # Clean nb_annees_experience (months → years)
    if 'nb_annees_experience' in df.columns:
        df['nb_annees_experience'] = df['nb_annees_experience'].apply(
            clean_nb_annees_experience
        )
        nb_cleaned = df['nb_annees_experience'].notna().sum()
        print(f"  ✓ nb_annees_experience cleaned: {nb_cleaned} records")
    
    # Replace experience_required with categories
    if 'experience_required' in df.columns:
        df['experience_required'] = df['experience_required'].apply(
            extract_experience_level
        )
        
        # Display distribution
        print("\n  Experience categories distribution:")
        for level, count in df['experience_required'].value_counts().items():
            pct = count / len(df) * 100
            print(f"    - {level}: {count} ({pct:.1f}%)")
    
    return df


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("\n🧪 Test des fonctions\n")
    
    # Test clean_nb_annees_experience
    print("1. Test clean_nb_annees_experience:")
    test_values = [18, 24, 36, 180, 5, None]
    for val in test_values:
        result = clean_nb_annees_experience(val)
        print(f"   {val} → {result}")
    
    # Test extract_experience_level
    print("\n2. Test extract_experience_level:")
    test_texts = [
            '18 Mois',
            '2 An(s)',
            '3 ans minimum',
            '5 ans minimum',
            'Expert (10 ans)',
            'Débutant accepté',
            None, 
            'Doctorat requis',
            'Aucune expérience',
            '3 An(s) - MMM, brand lift, attribut°, attent°', 
            '6 Mois - POSTE SIMILAIRE',
            'Expérience professionnelle avérée en gestion de projet, Expérience dans le domaine de l alimentation, santé et gaspillage alimentaire', 
            'mise en œuvre de programmes/projets, industries créatives, analyse et gestion des données, IA'

    ]
    for text in test_texts:
        result = extract_experience_level(text)
        print(f"   '{text}' → {result}")
    
    print("\n✅ Tests terminés")
