"""
Complete Job Market Analysis Pipeline - STAR SCHEMA VERSION
============================================================
Author: Ruche's teams
Date: 2026-01-05

Updates:
- Star Schema implementation (MotherDuck/DuckDB)
- Dimensional modeling with fact and dimension tables
- Snake case naming convention
- Proper foreign key relationships
"""

import os
import re
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter, defaultdict
from difflib import get_close_matches

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from dotenv import load_dotenv

# Stopwords
import nltk
from nltk.corpus import stopwords

# MongoDB
import pymongo
from pymongo import MongoClient

# MotherDuck
import duckdb
    # import Mduckdb name
from etl.config import MOTHERDUCK_DATABASE


# NLP & ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from tfidf_ml_data_filter import filter_data_jobs_ml 

# Configuration
warnings.filterwarnings('ignore')
load_dotenv()



# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# MongoDB Configuration
MONGO_URI = os.getenv('MONGO_URI')
DATABASE_NAME = "RUCHE_datalake"

# Configuration MotherDuck
MOTHERDUCK_TOKEN = os.getenv('MOTHERDUCK_TOKEN')
    # MOTHERDUCK_DATABASE in config.py

# Collections to process
COLLECTIONS = [
    "apec_raw",
    "francetravail_raw", 
    "servicepublic_raw",
    "jobteaser_raw"
]

# Testing limit (set to None for full dataset)
LIMIT = None

# NLP duplicate detection threshold
SIMILARITY_THRESHOLD = 0.9

# Output files
OUTPUT_RAW = 'job_offers_raw.xlsx'
OUTPUT_CLEANED = 'job_offers_cleaned.xlsx'
OUTPUT_DUPLICATES_XLSX = 'duplicates_high_similarity.xlsx'
OUTPUT_VISUALIZATION = 'duplicate_heatmap.png'

# Import french stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

FRENCH_STOPWORDS = list(stopwords.words('french'))


# ============================================================================
# ✅ AMÉLIORATION #1: MAPPING RÉGIONAL COMPLET
# ============================================================================

# Mapping complet département -> région pour toute la France
COMPLETE_REGION_MAPPING = {
    # Île-de-France
    '75': ('Île-de-France', '11'), '77': ('Île-de-France', '11'),
    '78': ('Île-de-France', '11'), '91': ('Île-de-France', '11'),
    '92': ('Île-de-France', '11'), '93': ('Île-de-France', '11'),
    '94': ('Île-de-France', '11'), '95': ('Île-de-France', '11'),
    
    # Auvergne-Rhône-Alpes
    '01': ('Auvergne-Rhône-Alpes', '84'), '03': ('Auvergne-Rhône-Alpes', '84'),
    '07': ('Auvergne-Rhône-Alpes', '84'), '15': ('Auvergne-Rhône-Alpes', '84'),
    '26': ('Auvergne-Rhône-Alpes', '84'), '38': ('Auvergne-Rhône-Alpes', '84'),
    '42': ('Auvergne-Rhône-Alpes', '84'), '43': ('Auvergne-Rhône-Alpes', '84'),
    '63': ('Auvergne-Rhône-Alpes', '84'), '69': ('Auvergne-Rhône-Alpes', '84'),
    '73': ('Auvergne-Rhône-Alpes', '84'), '74': ('Auvergne-Rhône-Alpes', '84'),
    
    # Provence-Alpes-Côte d'Azur
    '04': ("Provence-Alpes-Côte d'Azur", '93'), '05': ("Provence-Alpes-Côte d'Azur", '93'),
    '06': ("Provence-Alpes-Côte d'Azur", '93'), '13': ("Provence-Alpes-Côte d'Azur", '93'),
    '83': ("Provence-Alpes-Côte d'Azur", '93'), '84': ("Provence-Alpes-Côte d'Azur", '93'),
    
    # Nouvelle-Aquitaine
    '16': ('Nouvelle-Aquitaine', '75'), '17': ('Nouvelle-Aquitaine', '75'),
    '19': ('Nouvelle-Aquitaine', '75'), '23': ('Nouvelle-Aquitaine', '75'),
    '24': ('Nouvelle-Aquitaine', '75'), '33': ('Nouvelle-Aquitaine', '75'),
    '40': ('Nouvelle-Aquitaine', '75'), '47': ('Nouvelle-Aquitaine', '75'),
    '64': ('Nouvelle-Aquitaine', '75'), '79': ('Nouvelle-Aquitaine', '75'),
    '86': ('Nouvelle-Aquitaine', '75'), '87': ('Nouvelle-Aquitaine', '75'),
    
    # Occitanie
    '09': ('Occitanie', '76'), '11': ('Occitanie', '76'), '12': ('Occitanie', '76'),
    '30': ('Occitanie', '76'), '31': ('Occitanie', '76'), '32': ('Occitanie', '76'),
    '34': ('Occitanie', '76'), '46': ('Occitanie', '76'), '48': ('Occitanie', '76'),
    '65': ('Occitanie', '76'), '66': ('Occitanie', '76'), '81': ('Occitanie', '76'),
    '82': ('Occitanie', '76'),
    
    # Hauts-de-France
    '02': ('Hauts-de-France', '32'), '59': ('Hauts-de-France', '32'),
    '60': ('Hauts-de-France', '32'), '62': ('Hauts-de-France', '32'),
    '80': ('Hauts-de-France', '32'),
    
    # Grand Est
    '08': ('Grand Est', '44'), '10': ('Grand Est', '44'), '51': ('Grand Est', '44'),
    '52': ('Grand Est', '44'), '54': ('Grand Est', '44'), '55': ('Grand Est', '44'),
    '57': ('Grand Est', '44'), '67': ('Grand Est', '44'), '68': ('Grand Est', '44'),
    '88': ('Grand Est', '44'),
    
    # Bretagne
    '22': ('Bretagne', '53'), '29': ('Bretagne', '53'),
    '35': ('Bretagne', '53'), '56': ('Bretagne', '53'),
    
    # Pays de la Loire
    '44': ('Pays de la Loire', '52'), '49': ('Pays de la Loire', '52'),
    '53': ('Pays de la Loire', '52'), '72': ('Pays de la Loire', '52'),
    '85': ('Pays de la Loire', '52'),
    
    # Normandie
    '14': ('Normandie', '28'), '27': ('Normandie', '28'), '50': ('Normandie', '28'),
    '61': ('Normandie', '28'), '76': ('Normandie', '28'),
    
    # Bourgogne-Franche-Comté
    '21': ('Bourgogne-Franche-Comté', '27'), '25': ('Bourgogne-Franche-Comté', '27'),
    '39': ('Bourgogne-Franche-Comté', '27'), '58': ('Bourgogne-Franche-Comté', '27'),
    '70': ('Bourgogne-Franche-Comté', '27'), '71': ('Bourgogne-Franche-Comté', '27'),
    '89': ('Bourgogne-Franche-Comté', '27'), '90': ('Bourgogne-Franche-Comté', '27'),
    
    # Centre-Val de Loire
    '18': ('Centre-Val de Loire', '24'), '28': ('Centre-Val de Loire', '24'),
    '36': ('Centre-Val de Loire', '24'), '37': ('Centre-Val de Loire', '24'),
    '41': ('Centre-Val de Loire', '24'), '45': ('Centre-Val de Loire', '24'),
    
    # Corse
    '2A': ('Corse', '94'), '2B': ('Corse', '94'),
}

CITY_TO_DEPARTMENT = {
    # Paris (75)
    'paris': '75',
    
    # Lyon (69)
    'lyon': '69',
    
    # Marseille (13)
    'marseille': '13',
    
    # Autres grandes villes (pour référence)
    'lille': '59',
    'toulouse': '31',
    'nice': '06',
    'nantes': '44',
    'strasbourg': '67',
    'montpellier': '34',
    'bordeaux': '33',
    'rennes': '35',
    'reims': '51',
    'saint-étienne': '42',
    'toulon': '83',
    'grenoble': '38',
    'dijon': '21',
    'angers': '49',
    'nîmes': '30',
    'villeurbanne': '69',
    'clermont-ferrand': '63',
    'aix-en-provence': '13',
}

# ============================================================================
# PHASE 1: ETL PIPELINE - EXTRACTION & HARMONIZATION
# [Code ETL identique - non modifié pour brièveté]
# ============================================================================

def connect_mongodb(uri: str = MONGO_URI, db_name: str = DATABASE_NAME) -> pymongo.database.Database:
    """Establish connection to MongoDB Atlas"""
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.server_info()
        db = client[db_name]
        print("STEP 1.1: Connected to MongoDB: {}".format(db_name))
        return db
    except Exception as e:
        print("ERROR: MongoDB connection failed: {}".format(e))
        raise


def extract_collection(db: pymongo.database.Database, 
                       collection_name: str, 
                       limit: Optional[int] = None) -> List[Dict]:
    """Extract documents from a MongoDB collection"""
    collection = db[collection_name]
    total = collection.count_documents({})
    
    # ✅ DEBUG: Afficher le paramètre limit reçu
    print("🔍 DEBUG - limit parameter: {}".format(limit))
    
    print("STEP 1.2: Collection: {}".format(collection_name))
    print("  Total documents: {}".format(total))
    
    if limit:
        documents = list(collection.find().limit(limit))
        print("  Extracted: {} (limited)".format(len(documents)))
    else:
        # ✅ Vérifier qu'on arrive bien ici
        print("  ℹ️ No limit applied, fetching all...")
        documents = list(collection.find())
        print("  Extracted: {} (full)".format(len(documents)))
    
    return documents


def extract_all_collections(limit: Optional[int] = LIMIT) -> Dict[str, List[Dict]]:
    """Extract all collections from MongoDB"""
    print("=" * 80)
    print("PHASE 1: MONGODB EXTRACTION")
    print("=" * 80)
    
    db = connect_mongodb()
    
    all_data = {}
    for collection_name in COLLECTIONS:
        all_data[collection_name] = extract_collection(db, collection_name, limit)
    
    total_docs = sum(len(docs) for docs in all_data.values())
    print("STEP 1.3: Extraction complete: {} total documents".format(total_docs))
    
    return all_data


def extract_department_from_location(location: str) -> str:
    """Extract department number from location string"""
    if not location:
        return ''
    
    match = re.search(r'\b(\d{2,3})\b', location)
    if match:
        return match.group(1)
    
    # Corse cases 
    if 'corse' in location.lower():
        if '2a' in location.lower():
            return '2A'
        elif '2b' in location.lower():
            return '2B'
    
    return ''


def parse_date(date_str: str) -> str:
    """Parse and normalize dates to ISO format"""
    if not date_str or date_str == '':
        return ''
    
    if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
        return date_str
    
    match = re.match(r'(\d{2})/(\d{2})/(\d{4})', date_str)
    if match:
        day, month, year = match.groups()
        return f"{year}-{month}-{day}"
    
    return date_str


def harmonize_apec(doc: Dict) -> Dict:
    """Map APEC document to unified schema"""
    return {
        'job_id': doc.get('id', ''),
        'source_platform': 'APEC',
        'source_url': doc.get('url', ''),
        'scraped_at': doc.get('scraped_at', ''),
        'title': doc.get('title', ''),
        'description': doc.get('job_description', ''),
        'company_name': doc.get('company_name', ''),
        'company_description': doc.get('company_description', ''),
        'contract_type': doc.get('contract_type', ''),
        'location': doc.get('location', ''),
        'department': extract_department_from_location(doc.get('location', '')),
        'remote_work': '',
        'publication_date': parse_date(doc.get('publication_date', '')),
        'update_date': parse_date(doc.get('update_date', '')),
        'start_date': doc.get('start_date', ''),
        'application_deadline': '',
        'salary': doc.get('salary', ''),
        'benefits': '',
        'experience_required': doc.get('experience_required', ''),
        'experience_years': str(doc.get('experience_years', '')) if doc.get('experience_years') else '',
        'education_level': '',
        'job_grade': doc.get('status', ''),
        'hard_skills': doc.get('hard_skills', []),
        'soft_skills': doc.get('soft_skills', []),
        'languages': doc.get('languages', []),
        'certifications': [],
        'sector': doc.get('sector', ''),
        'job_function': doc.get('job_title', ''),
        'matched_keywords': [],
        'reference_external': doc.get('reference_apec', ''),
    }


def harmonize_francetravail(doc: Dict) -> Dict:
    """Map France Travail document to unified schema"""
    return {
        'job_id': doc.get('id', ''),
        'source_platform': 'France Travail',
        'source_url': doc.get('url_origine', ''),
        'scraped_at': doc.get('scraped_at', ''),
        'title': doc.get('intitule', ''),
        'description': doc.get('description', ''),
        'company_name': doc.get('entreprise', ''),
        'company_description': '',
        'contract_type': doc.get('type_contrat', ''),
        'location': doc.get('lieu', ''),
        'department': doc.get('lieu', '').split(' - ')[0] if ' - ' in doc.get('lieu', '') else '',
        'remote_work': '',
        'publication_date': doc.get('date_creation', ''),
        'update_date': doc.get('date_actualisation', ''),
        'start_date': '',
        'application_deadline': '',
        'salary': doc.get('salaire', ''),
        'benefits': '',
        'experience_required': doc.get('experience', ''),
        'experience_years': doc.get('experience', ''),
        'education_level': ', '.join(doc.get('formations', [])) if doc.get('formations') else '',
        'job_grade': '',
        'hard_skills': doc.get('competences', []),
        'soft_skills': [],
        'languages': doc.get('langues', []),
        'certifications': doc.get('permis', []),
        'sector': '',
        'job_function': '',
        'matched_keywords': doc.get('matched_keywords', []),
        'reference_external': '',
    }


def harmonize_servicepublic(doc: Dict) -> Dict:
    """Map Service Public document to unified schema"""
    return {
        'job_id': doc.get('id', ''),
        'source_platform': 'Service Public',
        'source_url': doc.get('url_origine', ''),
        'scraped_at': doc.get('scraped_at', ''),
        'title': doc.get('intitule', ''),
        'description': doc.get('description', ''),
        'company_name': doc.get('entreprise', ''),
        'company_description': '',
        'contract_type': doc.get('type_contrat', ''),
        'location': doc.get('lieu', ''),
        'department': doc.get('departement', ''),
        'remote_work': doc.get('teletravail', ''),
        'publication_date': doc.get('date_creation', ''),
        'update_date': '',
        'start_date': '',
        'application_deadline': doc.get('date_limite', ''),
        'salary': doc.get('salaire', ''),
        'benefits': doc.get('avantages', ''),
        'experience_required': doc.get('experience', ''),
        'experience_years': doc.get('experience', ''),
        'education_level': ', '.join(doc.get('formations', [])) if doc.get('formations') else '',
        'job_grade': doc.get('grade', ''),
        'hard_skills': doc.get('competences', []),
        'soft_skills': [],
        'languages': doc.get('langues', []),
        'certifications': doc.get('permis', []),
        'sector': '',
        'job_function': '',
        'matched_keywords': doc.get('matched_keywords', []),
        'reference_external': doc.get('mot_cle', ''),
    }


def harmonize_jobteaser(doc: Dict) -> Dict:
    """Map JobTeaser document to unified schema"""
    return {
        'job_id': doc.get('id', ''),
        'source_platform': 'JobTeaser',
        'source_url': doc.get('url', ''),
        'scraped_at': doc.get('scraped_at', ''),
        'title': doc.get('title', ''),
        'description': doc.get('description_clean', ''),
        'company_name': doc.get('company', ''),
        'company_description': '',
        'contract_type': doc.get('contract_raw', ''),
        'location': doc.get('location_raw', ''),
        'department': extract_department_from_location(doc.get('location_raw', '')),
        'remote_work': doc.get('remote_raw', ''),
        'publication_date': doc.get('publication_date', ''),
        'update_date': '',
        'start_date': doc.get('start_date_raw', ''),
        'application_deadline': doc.get('application_deadline', ''),
        'salary': doc.get('salary_raw', ''),
        'benefits': '',
        'experience_required': '',
        'experience_years': str(doc.get('experience_years', '')) if doc.get('experience_years') else '',
        'education_level': doc.get('education_level_raw', ''),
        'job_grade': '',
        'hard_skills': doc.get('hard_skills', []),
        'soft_skills': doc.get('soft_skills', []),
        'languages': doc.get('languages', []),
        'certifications': [],
        'sector': '',
        'job_function': doc.get('function_raw', ''),
        'matched_keywords': [],
        'reference_external': doc.get('search_keyword', ''),
    }


def harmonize_document(doc: Dict, collection_name: str) -> Dict:
    """Route document to appropriate harmonization function"""
    mapping_functions = {
        'apec_raw': harmonize_apec,
        'francetravail_raw': harmonize_francetravail,
        'servicepublic_raw': harmonize_servicepublic,
        'jobteaser_raw': harmonize_jobteaser,
    }
    
    harmonizer = mapping_functions.get(collection_name)
    if not harmonizer:
        raise ValueError("Unknown collection: {}".format(collection_name))
    
    return harmonizer(doc)


def harmonize_all_data(raw_data: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Harmonize all documents from all collections"""
    print("=" * 80)
    print("PHASE 2: HARMONIZATION & CLEANING")
    print("=" * 80)
    
    all_harmonized = []
    
    for collection_name, documents in raw_data.items():
        print("STEP 2.1: Harmonizing {}...".format(collection_name))
        
        for doc in documents:
            try:
                harmonized = harmonize_document(doc, collection_name)
                all_harmonized.append(harmonized)
            except Exception as e:
                print(" [!] ERROR: Error harmonizing document {}: {}".format(doc.get('id', 'unknown'), e))
    
    df = pd.DataFrame(all_harmonized)
    
    print("STEP 2.2: Harmonization complete: {} documents".format(len(df)))
    print("  Columns: {}".format(len(df.columns)))
    
    return df


def perform_eda(df: pd.DataFrame) -> None:
    """Perform Exploratory Data Analysis"""
    print("=" * 80)
    print("PHASE 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    print("STEP 3.1: Missing Values Analysis")
    print("-" * 60)
    
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    
    for col, pct in missing_pct.items():
        if pct > 0:
            print("  {}: {:.1f}%".format(col, pct))
    
    print("STEP 3.2: Distribution by Source Platform")
    print("-" * 60)
    
    source_counts = df['source_platform'].value_counts()
    total = len(df)
    
    for platform, count in source_counts.items():
        pct = count / total * 100
        print("  {}: {} ({:.1f}%)".format(platform, count, pct))
    
    print("STEP 3.3: Additional Statistics")
    print("-" * 60)
    print("  Total job offers: {}".format(len(df)))
    print("  Unique companies: {}".format(df['company_name'].nunique()))
    print("  Unique locations: {}".format(df['location'].nunique()))


def detect_duplicates_nlp(df: pd.DataFrame, 
                         threshold: float = SIMILARITY_THRESHOLD) -> pd.DataFrame:
    """Detect near-duplicate job offers using TF-IDF + Cosine Similarity"""
    print("=" * 80)
    print("PHASE 4: NLP DUPLICATE DETECTION")
    print("=" * 80)
    
    df_work = df.copy()
    
    print("STEP 4.1: Processing {} job descriptions...".format(len(df_work)))
    
    descriptions = df_work['description'].fillna('').astype(str)
    non_empty = descriptions.str.len() > 20
    
    print("  Valid descriptions: {}/{}".format(non_empty.sum(), len(descriptions)))
    
    if non_empty.sum() < 2:
        print("  Insufficient valid descriptions for duplicate detection")
        df['is_duplicate'] = False
        df['duplicate_group_id'] = None
        df['similarity_score'] = 0.0
        return df
    
    print("STEP 4.2: TF-IDF Vectorization (French stopwords)...")
    
    vectorizer = TfidfVectorizer(
        max_features=1000,
        max_df=0.8,
        min_df=2,
        stop_words=FRENCH_STOPWORDS,
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents='unicode'
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(descriptions[non_empty])
        print("  Matrix shape: {}".format(tfidf_matrix.shape))
        print("  Sparsity: {:.1f}%".format((1 - tfidf_matrix.nnz / np.prod(tfidf_matrix.shape)) * 100))
    except Exception as e:
        print("  Vectorization failed: {}".format(e))
        df['is_duplicate'] = False
        df['duplicate_group_id'] = None
        df['similarity_score'] = 0.0
        return df
    
    print("STEP 4.3: Computing pairwise similarities (threshold: {})...".format(threshold))
    
    n_docs = tfidf_matrix.shape[0]
    duplicates_found = []
    
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    for i in range(n_docs):
        for j in range(i + 1, n_docs):
            if similarity_matrix[i, j] >= threshold:
                duplicates_found.append((i, j, similarity_matrix[i, j]))
    
    print("  Found {} similar pairs".format(len(duplicates_found)))
    
    print("STEP 4.4: Assigning duplicate groups...")
    
    df_work['is_duplicate'] = False
    df_work['duplicate_group_id'] = None
    df_work['similarity_score'] = 0.0
    
    if duplicates_found:
        graph = defaultdict(set)
        for i, j, score in duplicates_found:
            graph[i].add(j)
            graph[j].add(i)
        
        visited = set()
        group_id = 0
        
        def dfs(node, group):
            visited.add(node)
            group.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, group)
        
        for node in graph:
            if node not in visited:
                group = set()
                dfs(node, group)
                
                for idx in group:
                    actual_idx = df_work.index[non_empty][idx]
                    df_work.loc[actual_idx, 'is_duplicate'] = True
                    df_work.loc[actual_idx, 'duplicate_group_id'] = f"DUP_{group_id}"
                
                group_id += 1
        
        for i, j, score in duplicates_found:
            actual_i = df_work.index[non_empty][i]
            actual_j = df_work.index[non_empty][j]
            
            df_work.loc[actual_i, 'similarity_score'] = max(
                df_work.loc[actual_i, 'similarity_score'], score
            )
            df_work.loc[actual_j, 'similarity_score'] = max(
                df_work.loc[actual_j, 'similarity_score'], score
            )
        
        print("  Created {} duplicate groups".format(group_id))
        print("  Total duplicates: {}".format(df_work['is_duplicate'].sum()))
    
    print("STEP 4.5: Duplicate Detection Summary")
    print("  Total offers analyzed: {}".format(len(df_work)))
    print("  Duplicates found: {} ({:.1f}%)".format(
        df_work['is_duplicate'].sum(),
        df_work['is_duplicate'].sum()/len(df_work)*100
    ))
    print("  Unique offers: {}".format(len(df_work) - df_work['is_duplicate'].sum()))
    
    return df_work


def create_unified_dataset(limit: Optional[int] = LIMIT) -> pd.DataFrame:
    """Complete ETL pipeline: Extract + Harmonize + Detect Duplicates"""
    raw_data = extract_all_collections(limit)
    df = harmonize_all_data(raw_data)
    perform_eda(df)
    df = detect_duplicates_nlp(df, threshold=SIMILARITY_THRESHOLD)
    
    print("=" * 80)
    print("ETL PIPELINE COMPLETE")
    print("=" * 80)
    print("Dataset shape: {}".format(df.shape))
    
    return df


# ============================================================================
# PHASE 2: DATA CLEANING & STANDARDIZATION
# [Code nettoyage identique - non modifié pour brièveté]
# ============================================================================

# [Inclure toutes les fonctions de nettoyage...]
# Pour brièveté, je vais juste inclure les signatures essentielles

def visualize_duplicates(df, output_heatmap=OUTPUT_VISUALIZATION, output_xlsx=OUTPUT_DUPLICATES_XLSX):
    """Create heatmap visualization and export duplicates"""
    print("STEP 5.1: Duplicate visualization and export")
    # [Code original conservé]
    pass

def clean_scraped_at_date(df):
    """Convert scraped_at to date only"""
    print("STEP 5.2: Scraped_at date cleaning")
    # [Code original conservé]
    return df

def filter_and_clean_titles(df):
    """Filter unwanted titles and standardize capitalization"""
    print("STEP 5.3: Title filtering and cleaning")
    # [Code original conservé]
    return df

def standardize_contract_types(df):
    """Standardize contract_type values"""
    print("STEP 5.4: Contract type standardization")
    # [Code original conservé]
    return df

def enhanced_location_cleaning(df):
    """Advanced location cleaning with arrondissement handling"""
    print("STEP 5.5: Enhanced location cleaning")
    # [Code original conservé]
    return df

def enhanced_department_extraction(df):
    """Extract department codes from location"""
    print("STEP 5.6: Enhanced department extraction")
    # [Code original conservé]
    return df

def recategorize_remote_work(df):
    """Recategorize remote_work into 3 categories"""
    print("STEP 5.7: Remote work recategorization")
    # [Code original conservé]
    return df

def clean_application_deadline(df):
    """Clean and format application_deadline"""
    print("STEP 5.8: Application deadline cleaning")
    # [Code original conservé]
    return df

def enhanced_experience_required_cleaning(df):
    """Enhanced cleaning for experience_required"""
    print("STEP 5.9: Enhanced experience required cleaning")
    # [Code original conservé]
    return df

def recode_job_function(df):
    """Recode job_function: Non classe -> Autre"""
    print("STEP 5.10: Job function recoding")
    # [Code original conservé]
    return df

def deduplicate_offers(df):
    """Remove duplicates keeping only the best record per group"""
    print("STEP 6.1: Deduplication")
    # [Code original conservé]
    return df

def standardize_job_functions(df):
    """Standardize job_function using APEC nomenclature"""
    print("STEP 6.2: Job function standardization")
    # [Code original conservé]
    return df

def infer_job_functions_universal(df):
    """Infer job_function for ALL records"""
    print("STEP 6.3: Universal job function inference")
    # [Code original conservé]
    return df

def filter_data_jobs(df):
    """Filter to keep only genuine data-related jobs - ML ENHANCED"""
    print("STEP 6.4: Data job filtering (ML-based)")
    return filter_data_jobs_ml(df, FRENCH_STOPWORDS)

def standardize_company_names(df):
    """Standardize company names"""
    print("STEP 6.5: Company name standardization")
    # [Code original conservé]
    return df

def extract_contract_duration(df):
    """Extract duration from contract_type"""
    print("STEP 6.6: Contract duration extraction")
    # [Code original conservé]
    return df

def clean_update_date(df):
    """Convert update_date to date format"""
    print("STEP 6.7: Update date cleaning")
    # [Code original conservé]
    return df

def clean_salary(df):
    """Standardize salary to annual gross ranges"""
    print("STEP 6.8: Salary cleaning")
    # [Code original conservé]
    return df

def clean_experience_years(df):
    """Clean experience_years: extract numeric values only"""
    print("STEP 6.9: Experience years cleaning")
    # [Code original conservé]
    return df

def standardize_skills_and_languages(df):
    """Standardize hard_skills, soft_skills, and languages"""
    print("STEP 6.10: Skills and languages standardization")
    # [Code original conservé]
    return df

def rename_to_driving_license(df):
    """Rename certifications to driving_license"""
    print("STEP 6.11: Driving license conversion")
    # [Code original conservé]
    return df


def clean_job_data(df: pd.DataFrame, output_file: str = 'job_offers_cleaned.xlsx') -> pd.DataFrame:
    """
    Execute complete data cleaning pipeline with granular logging
    
    ✅ CORRECTIONS APPLIQUÉES:
    - Logging après chaque étape
    - Identification des suppressions massives
    - Alertes si perte > 20%
    """
    print("=" * 80)
    print("PHASE 5: DATA CLEANING PIPELINE - START (FIXED)")
    print("=" * 80)
    
    # ✅ CORRECTION: Tracker initial
    initial_count = len(df)
    print("📊 Initial records: {}".format(initial_count))
    
    # Import des fonctions (à adapter selon votre structure)
    from etl_mongo_mduck import (
        visualize_duplicates,
        clean_scraped_at_date,
        filter_and_clean_titles,
        standardize_contract_types,
        enhanced_department_extraction,
        enhanced_location_cleaning,
        recategorize_remote_work,
        clean_application_deadline,
        deduplicate_offers,
        standardize_job_functions,
        infer_job_functions_universal,
        filter_data_jobs,
        standardize_company_names,
        extract_contract_duration,
        clean_update_date,
        clean_salary,
        enhanced_experience_required_cleaning,
        clean_experience_years,
        standardize_skills_and_languages,
        rename_to_driving_license,
        recode_job_function
    )
    
    # NEW cleaning steps
    visualize_duplicates(df)
    
    df = clean_scraped_at_date(df)
    print("  → After date cleaning: {} (-{})".format(len(df), initial_count - len(df)))
    
    df = filter_and_clean_titles(df)
    after_titles = len(df)
    loss_titles = initial_count - after_titles
    print("  → After title filtering: {} (-{}, {:.1f}%)".format(
        after_titles, loss_titles, loss_titles / initial_count * 100
    ))
    if loss_titles / initial_count > 0.2:
        print("  ⚠️ WARNING: Title filtering removed >20% of data!")
    
    df = standardize_contract_types(df)
    df = enhanced_department_extraction(df)
    df = enhanced_location_cleaning(df)
    df = recategorize_remote_work(df)
    df = clean_application_deadline(df)
    
    # ⚠️ POINT CRITIQUE: Deduplication
    before_dedup = len(df)
    df = deduplicate_offers(df)
    after_dedup = len(df)
    loss_dedup = before_dedup - after_dedup
    print("  ⚠️ Deduplication removed: {} records ({:.1f}%)".format(
        loss_dedup,
        loss_dedup / before_dedup * 100
    ))
    if loss_dedup / before_dedup > 0.3:
        print("  🚨 ALERT: Deduplication removed >30% of data! Check similarity threshold.")
    
    # Existing cleaning steps
    df = standardize_job_functions(df)
    df = infer_job_functions_universal(df)
    
    # ⚠️ POINT CRITIQUE: ML filtering
    before_ml = len(df)
    df = filter_data_jobs(df)
    after_ml = len(df)
    loss_ml = before_ml - after_ml
    print("  ⚠️ ML filtering removed: {} records ({:.1f}%)".format(
        loss_ml,
        loss_ml / before_ml * 100
    ))
    if loss_ml / before_ml > 0.4:
        print("  🚨 ALERT: ML filtering removed >40% of data! Check filter_data_jobs_ml parameters.")
    
    df = standardize_company_names(df)
    df = extract_contract_duration(df)
    df = clean_update_date(df)
    df = clean_salary(df)
    df = enhanced_experience_required_cleaning(df)
    df = clean_experience_years(df)
    df = standardize_skills_and_languages(df)
    df = rename_to_driving_license(df)
    df = recode_job_function(df)
    
    # Export to Excel
    print("=" * 80)
    print("PHASE 6: EXPORTING CLEANED DATA")
    print("=" * 80)
    
    df.to_excel(output_file, index=False, engine='openpyxl')
    print("STEP 6.12: Cleaned data exported to: {}".format(output_file))
    
    # ✅ Summary with alerts
    final_count = len(df)
    total_loss = initial_count - final_count
    retention_rate = final_count / initial_count * 100
    
    print("\n📊 CLEANING SUMMARY:")
    print("  • Initial records: {}".format(initial_count))
    print("  • Final records: {}".format(final_count))
    print("  • Total removed: {} ({:.1f}%)".format(total_loss, 100 - retention_rate))
    print("  • Retention rate: {:.1f}%".format(retention_rate))
    
    if retention_rate < 60:
        print("\n  🚨 CRITICAL: <60% retention! Review filtering logic.")
    elif retention_rate < 80:
        print("\n  ⚠️ WARNING: <80% retention. Consider loosening filters.")
    else:
        print("\n  ✅ Good: Retention >80%")
    
    print("=" * 80)
    print("DATA CLEANING PIPELINE - COMPLETE")
    print("=" * 80)
    
    return df


# ============================================================================
# PHASE 3: STAR SCHEMA MODELING - MOTHERDUCK/DUCKDB
# ============================================================================

def connect_motherduck(token: str = MOTHERDUCK_TOKEN, 
                       database: str = MOTHERDUCK_DATABASE) -> duckdb.DuckDBPyConnection:
    """Connect to MotherDuck cloud database with fallback to local"""
    print("=" * 80)
    print("PHASE 7: MOTHERDUCK CONNECTION")
    print("=" * 80)
    
    try:
        if not token:
            raise ValueError("MOTHERDUCK_TOKEN not found in environment")
        
        print("STEP 7.1: Connecting to MotherDuck: {}...".format(database))
        con = duckdb.connect("md:?motherduck_token={}".format(token))
        
        con.execute("CREATE DATABASE IF NOT EXISTS {}".format(database))
        
        con.close()
        con = duckdb.connect("md:{}?motherduck_token={}".format(database, token))
        
        print("STEP 7.2: Connected to MotherDuck: {}".format(database))
        return con
        
    except Exception as e:
        print("WARNING: MotherDuck connection failed: {}".format(e))
        print("STEP 7.3: Falling back to local DuckDB...")
        
        local_db = "local_job_market.duckdb"
        con = duckdb.connect(local_db)
        print("STEP 7.4: Connected to local DuckDB: {}".format(local_db))
        return con


def create_star_schema_ddl(con: duckdb.DuckDBPyConnection) -> None:
    """
    Create Star Schema tables with proper dimensional modeling
    
    Star Schema Structure:
    - F_Offre: Fact table (job offers)
    - D_Localisation: Location dimension
    - H_Region: Region hierarchy
    - D_Date: Date dimension
    - D_Contrat: Contract type dimension
    """
    print("=" * 80)
    print("PHASE 8: STAR SCHEMA CREATION")
    print("=" * 80)
    
    print("STEP 8.1: Creating dimension tables...")
    
    # ========================================================================
    # H_Region: Region Hierarchy Dimension
    # ========================================================================
    
    ddl_h_region = """
    CREATE OR REPLACE TABLE h_region (
        id_region INTEGER PRIMARY KEY,
        nom_region TEXT NOT NULL,
        code_region TEXT
    );
    """
    
    con.execute(ddl_h_region)
    print(" - Table created: h_region")
    
    # ========================================================================
    # D_Localisation: Location Dimension
    # ========================================================================
    
    ddl_d_localisation = """
    CREATE OR REPLACE TABLE d_localisation (
        id_ville INTEGER PRIMARY KEY,
        ville TEXT NOT NULL,
        code_postal TEXT,
        departement TEXT,
        latitude DOUBLE,
        longitude DOUBLE,
        id_region INTEGER,
        FOREIGN KEY (id_region) REFERENCES h_region(id_region)
    );
    """
    
    con.execute(ddl_d_localisation)
    print(" - Table created: d_localisation")
    
    # ========================================================================
    # D_Date: Date Dimension
    # ========================================================================
    
    ddl_d_date = """
    CREATE OR REPLACE TABLE d_date (
        id_date INTEGER PRIMARY KEY,
        date_complete DATE,
        jour INTEGER,
        mois INTEGER,
        annee INTEGER,
        trimestre INTEGER,
        nom_mois TEXT,
        nom_jour TEXT,
        semaine INTEGER,
        jour_annee INTEGER
    );
    """
    
    con.execute(ddl_d_date)
    print(" - Table created: d_date")
    
    # ========================================================================
    # D_Contrat: Contract Type Dimension
    # ========================================================================
    
    ddl_d_contrat = """
    CREATE OR REPLACE TABLE d_contrat (
        id_contrat INTEGER PRIMARY KEY,
        type_contrat TEXT NOT NULL,
        is_cdi BOOLEAN DEFAULT FALSE,
        is_cdd BOOLEAN DEFAULT FALSE,
        is_interim BOOLEAN DEFAULT FALSE,
        is_stage BOOLEAN DEFAULT FALSE,
        is_apprentissage BOOLEAN DEFAULT FALSE,
        is_freelance BOOLEAN DEFAULT FALSE,
        duree_mois INTEGER,
        description_contrat TEXT
    );
    """
    
    con.execute(ddl_d_contrat)
    print(" - Table created: d_contrat")
    
    print("STEP 8.2: Creating fact table...")
    
    # ========================================================================
    # F_Offre: Fact Table (Job Offers)
    # ========================================================================
    
    ddl_f_offre = """
    CREATE OR REPLACE TABLE f_offre (
        job_id TEXT PRIMARY KEY,
        source_platform TEXT NOT NULL,
        source_url TEXT,
        scraped_at DATE,
        title TEXT NOT NULL,
        description TEXT,
        company_name TEXT,
        company_description TEXT,
        nb_annees_experience INTEGER,
        experience_required TEXT,
        id_ville INTEGER,
        id_region INTEGER,
        id_contrat INTEGER,
        id_date_publication INTEGER,
        id_date_deadline INTEGER,
        is_teletravail BOOLEAN DEFAULT FALSE,
        avantages TEXT,
        salaire TEXT,
        hard_skills TEXT,
        soft_skills TEXT,
        langages TEXT,
        education_level TEXT,
        job_function TEXT,
        sector TEXT,
        job_grade TEXT,
        driving_license BOOLEAN DEFAULT FALSE,
        is_duplicate BOOLEAN DEFAULT FALSE,
        similarity_score DOUBLE,
        FOREIGN KEY (id_ville) REFERENCES d_localisation(id_ville),
        FOREIGN KEY (id_region) REFERENCES h_region(id_region),
        FOREIGN KEY (id_contrat) REFERENCES d_contrat(id_contrat),
        FOREIGN KEY (id_date_publication) REFERENCES d_date(id_date),
        FOREIGN KEY (id_date_deadline) REFERENCES d_date(id_date)
    );
    """
    
    con.execute(ddl_f_offre)
    print("  Table created: f_offre")
    
    print("STEP 8.3: Star Schema DDL execution complete")
    print("  Dimension tables: 4 (h_region, d_localisation, d_date, d_contrat)")
    print("  Fact tables: 1 (f_offre)")


def populate_dimension_tables(df: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> dict:
    """Populate dimension tables from cleaned DataFrame"""
    
    print("=" * 80)
    print("PHASE 9: POPULATING DIMENSION TABLES")
    print("=" * 80)
    
    stats = {
        'h_region': 0, 
        'd_localisation': 0, 
        'd_date': 0, 
        'd_contrat': 0
    }
    
    # ========================================================================
    # IMPORTS
    # ========================================================================
    from clean_localisation import extract_city_from_location
    from geolocation_enrichment import GeoRefFranceV2, COMPLETE_REGION_MAPPING
    import time
    
    # ========================================================================
    # STEP 9.1: h_region EN PREMIER (toutes les régions)
    # ========================================================================
    
    print("STEP 9.1: Populating h_region (FIRST)...")
    
    # Extraire régions uniques depuis COMPLETE_REGION_MAPPING
    unique_regions = {}
    for dept, (region_name, region_code) in COMPLETE_REGION_MAPPING.items():
        if region_name not in unique_regions:
            unique_regions[region_name] = region_code
    
    region_data = [{
        'id_region': 0,
        'nom_region': 'UNKNOWN',
        'code_region': '00'
    }]
    
    for idx, (region_name, region_code) in enumerate(sorted(unique_regions.items()), 1):
        region_data.append({
            'id_region': idx,
            'nom_region': region_name,
            'code_region': region_code
        })
    
    df_regions = pd.DataFrame(region_data)
    con.execute("DELETE FROM h_region")
    con.execute("INSERT INTO h_region SELECT * FROM df_regions")
    
    stats['h_region'] = len(df_regions)
    print("  ✅ Inserted {} regions\n".format(len(df_regions)))
    
    # ========================================================================
    # STEP 9.2: d_localisation ENSUITE (avec géocodage)
    # ========================================================================
    
    print("STEP 9.2: Populating d_localisation...")
    
    # Extraire localisations uniques
    locations = df[['location']].copy()
    locations['location'] = locations['location'].fillna('UNKNOWN')
    locations = locations.drop_duplicates()
    
    print("  → {} unique locations found".format(len(locations)))
    
    # Nettoyer les villes
    print("  → Cleaning city names...")
    locations['ville_clean'] = locations['location'].apply(extract_city_from_location)
    print("  → {} unique cleaned cities".format(locations['ville_clean'].nunique()))
    
    # Dédoublonner sur ville_clean
    locations = locations.drop_duplicates(subset=['ville_clean'])
    # Géocoder avec API
    print("  → Geocoding with API (this may take 10-20 minutes)...\n")
    
    geo_api = GeoRefFranceV2()
    location_data = [{
        'id_ville': 0,
        'ville': 'UNKNOWN',
        'code_postal': '00',
        'departement': '00',
        'latitude': None,
        'longitude': None,
        'id_region': 0
    }]
    
    enriched = 0
    errors = 0
    
    for idx, row in enumerate(locations.itertuples(), 1):
        ville_clean = locations.loc[row.Index, 'ville_clean']
        
        if idx % 50 == 0:
            print("    Progress: {}/{} ({:.1f}%)".format(idx, len(locations), idx/len(locations)*100))
        
        if ville_clean == 'UNKNOWN':
            continue
        
        try:
            # Géocodage API
            result = geo_api.get_full_location_info(ville_clean)
            
            if result:
                lat, lon, dept_api = result
                
                # Trouver id_region (h_region existe déjà ✅)
                id_region = 0
                if dept_api in COMPLETE_REGION_MAPPING:
                    region_name, _ = COMPLETE_REGION_MAPPING[dept_api]
                    region_row = df_regions[df_regions['nom_region'] == region_name]
                    if not region_row.empty:
                        id_region = int(region_row.iloc[0]['id_region'])
                
                location_data.append({
                    'id_ville': idx,
                    'ville': ville_clean,
                    'code_postal': dept_api,
                    'departement': dept_api,  # ← Depuis API
                    'latitude': lat,
                    'longitude': lon,
                    'id_region': id_region   # ← Trouvé via MAPPING
                })
                enriched += 1
            else:
                location_data.append({
                    'id_ville': idx,
                    'ville': ville_clean,
                    'code_postal': '00',
                    'departement': '00',
                    'latitude': None,
                    'longitude': None,
                    'id_region': 0
                })
                errors += 1
            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            errors += 1
            location_data.append({
                'id_ville': idx,
                'ville': ville_clean,
                'code_postal': '00',
                'departement': '00',
                'latitude': None,
                'longitude': None,
                'id_region': 0
            })
    
    df_locations = pd.DataFrame(location_data)
    
    print("\n  Geocoding summary:")
    print("    - Total: {}".format(len(df_locations)))
    print("    - Enriched: {}".format(enriched))
    print("    - With valid region: {}".format((df_locations['id_region'] > 0).sum()))
    print("    - Errors: {}\n".format(errors))
    
    # Insertion
    con.execute("DELETE FROM d_localisation")
    con.execute("INSERT INTO d_localisation SELECT * FROM df_locations")
    stats['d_localisation'] = len(df_locations)
    print("  ✅ Inserted {} locations\n".format(len(df_locations)))
    
    # ========================================================================
    # STEP 9.3: d_date
    # ========================================================================
    
    print("STEP 9.3: Populating d_date...")
    
    dates_series = pd.to_datetime(df['publication_date'], errors='coerce').dropna()
    from datetime import date as dt_date
    
    date_data = [{
        'id_date': 0,
        'date_complete': dt_date(1900, 1, 1),
        'jour': 0,
        'mois': 0,
        'annee': 0,
        'trimestre': 0,
        'nom_mois': 'UNKNOWN',
        'nom_jour': 'UNKNOWN',
        'semaine': 0,
        'jour_annee': 0
    }]

    if len(dates_series) > 0:
        min_date = dates_series.min()
        max_date = dates_series.max()
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        for idx, date in enumerate(date_range, 1):
            date_data.append({
                'id_date': idx,
                'date_complete': date.date(),
                'jour': date.day,
                'mois': date.month,
                'annee': date.year,
                'trimestre': (date.month - 1) // 3 + 1,
                'nom_mois': date.strftime('%B'),
                'nom_jour': date.strftime('%A'),
                'semaine': date.isocalendar()[1],
                'jour_annee': date.timetuple().tm_yday
            })
        
        print("  Date range: {} to {}".format(
            min_date.strftime('%Y-%m-%d'),
            max_date.strftime('%Y-%m-%d')
        ))
    
    df_dates = pd.DataFrame(date_data)
    con.execute("DELETE FROM d_date")
    con.execute("INSERT INTO d_date SELECT * FROM df_dates")
    stats['d_date'] = len(df_dates)
    print("  ✅ Inserted {} dates\n".format(len(df_dates)))

    # ========================================================================
    # STEP 9.4: d_contrat
    # ========================================================================
    
    print("STEP 9.4: Populating d_contrat...")
    
    contract_types = df['contract_type'].fillna('UNKNOWN').replace('', 'UNKNOWN').unique()
    
    contract_data = [{
        'id_contrat': 0,
        'type_contrat': 'UNKNOWN',
        'is_cdi': False,
        'is_cdd': False,
        'is_interim': False,
        'is_stage': False,
        'is_apprentissage': False,
        'is_freelance': False,
        'duree_mois': None,
        'description_contrat': 'Type de contrat non renseigné'
    }]
    
    for idx, contract in enumerate(contract_types, 1):
        if contract == 'UNKNOWN':
            continue
        
        contract_str = str(contract).upper()
        
        contract_data.append({
            'id_contrat': idx,
            'type_contrat': contract,
            'is_cdi': 'CDI' in contract_str,
            'is_cdd': 'CDD' in contract_str,
            'is_interim': 'INTERIM' in contract_str or 'INTÉRIM' in contract_str,
            'is_stage': 'STAGE' in contract_str,
            'is_apprentissage': 'APPRENTISSAGE' in contract_str,
            'is_freelance': 'FREELANCE' in contract_str or 'INDÉPENDANT' in contract_str,
            'duree_mois': None,
            'description_contrat': contract
        })
    
    df_contracts = pd.DataFrame(contract_data)
    con.execute("DELETE FROM d_contrat")
    con.execute("INSERT INTO d_contrat SELECT * FROM df_contracts")
    stats['d_contrat'] = len(df_contracts)
    print("  ✅ Inserted {} contract types\n".format(len(df_contracts)))
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 80)
    print("DIMENSION TABLES POPULATION COMPLETE")
    print("=" * 80)
    for table, count in stats.items():
        print("  • {}: {} records".format(table, count))
    print("=" * 80 + "\n")
    
    return stats


def populate_fact_table(df: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> Tuple[int, Dict[str, int]]:
    """Populate fact table with job offers and foreign keys"""
    print("=" * 80)
    print("PHASE 10: POPULATING FACT TABLE")
    print("=" * 80)
    
    print("STEP 10.1: Preparing fact table data...")
    
    # Get dimension mappings
    df_locations = con.execute("SELECT * FROM d_localisation").fetchdf()
    df_contracts = con.execute("SELECT * FROM d_contrat").fetchdf()
    df_dates = con.execute("SELECT * FROM d_date").fetchdf()
    
    # Create location mapping
    location_map = df_locations.set_index('ville')['id_ville'].to_dict()
    location_map['UNKNOWN'] = 0
    location_map[''] = 0
    location_map[None] = 0

    # Create contract mapping
    contract_map = df_contracts.set_index('type_contrat')['id_contrat'].to_dict()
    contract_map['UNKNOWN'] = 0
    contract_map[''] = 0
    contract_map[None] = 0

    # Create date mapping (date -> id_date)
    date_map = df_dates.set_index('date_complete')['id_date'].to_dict()
    date_map[None] = 0

    print(" v Loaded {} location mappings".format(len(location_map)))
    print(" v Loaded {} contract mappings".format(len(contract_map)))
    print(" v Loaded {} date mappings".format(len(date_map)))

    print("STEP 10.2: Mapping foreign keys...")
    
    validation_stats = {
        'total_records': len(df),
        'valid_location': 0,
        'missing_location': 0,
        'valid_contract': 0,
        'missing_contract': 0,
        'valid_pub_date': 0,
        'invalid_pub_date': 0
    }

    # ✅ HELPER FUNCTION - Définie AVANT la boucle
    def safe_list_to_str(value):
        """Convert list/array to string safely"""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ''
        if isinstance(value, (list, tuple)):
            return ', '.join(str(v) for v in value if v)
        return str(value)

    # Prepare fact table DataFrame
    fact_data = []
    
    for idx, row in df.iterrows():
        # Map location
        location_value = row['location']
        if pd.isna(location_value) or location_value == '':
            location_value = 'UNKNOWN'
            validation_stats['missing_location'] += 1
        else:
            validation_stats['valid_location'] += 1
        
        id_ville = location_map.get(location_value, 0)
        
        # Map contract
        contract_value = row['contract_type']
        if pd.isna(contract_value) or contract_value == '':
            contract_value = 'UNKNOWN'
            validation_stats['missing_contract'] += 1
        else:
            validation_stats['valid_contract'] += 1
        
        id_contrat = contract_map.get(contract_value, 0)

        # Map publication date
        try:
            if pd.notna(row['publication_date']):
                pub_date = pd.to_datetime(row['publication_date']).date()
                id_date_publication = date_map.get(pub_date, 0)
                if id_date_publication != 0:
                    validation_stats['valid_pub_date'] += 1
                else:
                    validation_stats['invalid_pub_date'] += 1
            else:
                id_date_publication = 0
                validation_stats['invalid_pub_date'] += 1
        except:
            id_date_publication = 0
            validation_stats['invalid_pub_date'] += 1
        
        # Map deadline date
        try:
            if pd.notna(row['application_deadline']):
                deadline_date = pd.to_datetime(row['application_deadline']).date()
                id_date_deadline = date_map.get(deadline_date, 0)
            else:
                id_date_deadline = 0
        except:
            id_date_deadline = 0
        
        # Region lookup
        try:
            location_row = df_locations[df_locations['id_ville'] == id_ville]
            if not location_row.empty:
                id_region = location_row.iloc[0]['id_region']
            else:
                id_region = 0
        except:
            id_region = 0

        # Experience years
        try:
            exp_years = row['experience_years']
            if pd.notna(exp_years):
                import re
                match = re.search(r'\d+', str(exp_years))
                nb_annees_exp = int(match.group()) if match else None
            else:
                nb_annees_exp = None
        except:
            nb_annees_exp = None
        
        # ✅ CORRECTION: Convert skills using helper function
        hard_skills_text = safe_list_to_str(row.get('hard_skills'))
        soft_skills_text = safe_list_to_str(row.get('soft_skills'))
        languages_text = safe_list_to_str(row.get('languages'))
        
        # Remote work boolean
        remote_value = str(row.get('remote_work', '')).lower()
        is_teletravail = remote_value in ['oui', 'yes', 'true', 'partiel', 'total']
        
        # Driving license boolean
        driving_value = str(row.get('driving_license', '')).lower()
        has_driving_license = driving_value in ['yes', 'oui', 'true', 'permis b', 'permi b']

        fact_data.append({
            'job_id': row['job_id'],
            'source_platform': row['source_platform'],
            'source_url': row['source_url'],
            'scraped_at': pd.to_datetime(row['scraped_at']).date() if pd.notna(row['scraped_at']) else None,
            'title': row['title'],
            'description': row['description'],
            'company_name': row['company_name'],
            'company_description': row.get('company_description', ''),
            'nb_annees_experience': nb_annees_exp,
            'experience_required': row['experience_required'],
            'id_ville': id_ville,
            'id_region': id_region,
            'id_contrat': id_contrat,
            'id_date_publication': id_date_publication,
            'id_date_deadline': id_date_deadline,
            'is_teletravail': is_teletravail,
            'avantages': row.get('benefits', ''),
            'salaire': row.get('salary', ''),
            'hard_skills': hard_skills_text,
            'soft_skills': soft_skills_text,
            'langages': languages_text,
            'education_level': row.get('education_level', ''),
            'job_function': row.get('job_function', ''),
            'sector': row.get('sector', ''),
            'job_grade': row.get('job_grade', ''),
            'driving_license': has_driving_license,
            'is_duplicate': row.get('is_duplicate', False),
            'similarity_score': row.get('similarity_score', 0.0)
        })
    
    df_fact = pd.DataFrame(fact_data)
    
    print("STEP 10.3: Validation summary:")
    print("  • Total records: {}".format(validation_stats['total_records']))
    print("  • Valid locations: {}".format(validation_stats['valid_location']))
    print("  • Missing locations → UNKNOWN: {}".format(validation_stats['missing_location']))
    print("  • Valid contracts: {}".format(validation_stats['valid_contract']))
    print("  • Missing contracts → UNKNOWN: {}".format(validation_stats['missing_contract']))
    print("  • Valid publication dates: {}".format(validation_stats['valid_pub_date']))
    print("  • Invalid dates → UNKNOWN: {}".format(validation_stats['invalid_pub_date']))
    
    print("\nSTEP 10.4: Inserting {} records into f_offre...".format(len(df_fact)))
    con.execute("DELETE FROM f_offre")
    con.execute("INSERT INTO f_offre SELECT * FROM df_fact")
    
    # Verify
    count = con.execute("SELECT COUNT(*) FROM f_offre").fetchone()[0]
    print("  ✅ Verified: {} records in f_offre".format(count))
    
    if count != len(df_fact):
        print("  ⚠️ WARNING: Expected {} but got {} records!".format(len(df_fact), count))
    
    print("\n✅ STEP 10.5: Fact table population complete")
    
    # ✅ CORRECTION: Return tuple
    return count, validation_stats

def run_data_quality_checks(con: duckdb.DuckDBPyConnection) -> None:
    """
    ✅ NEW: Run data quality validation queries
    """
    print("=" * 80)
    print("PHASE 11: DATA QUALITY VALIDATION")
    print("=" * 80)
    
    checks = {
        "1. Total Records": "SELECT COUNT(*) as total FROM f_offre",
        
        "2. Records with UNKNOWN Location": """
            SELECT COUNT(*) as unknown_loc
            FROM f_offre f
            JOIN d_localisation l ON f.id_ville = l.id_ville
            WHERE l.ville = 'UNKNOWN'
        """,
        
        "3. Records with UNKNOWN Contract": """
            SELECT COUNT(*) as unknown_contract
            FROM f_offre f
            JOIN d_contrat c ON f.id_contrat = c.id_contrat
            WHERE c.type_contrat = 'UNKNOWN'
        """,
        
        "4. Records with UNKNOWN Date": """
            SELECT COUNT(*) as unknown_date
            FROM f_offre
            WHERE id_date_publication = 0
        """,
        
        "5. Distribution by Platform": """
            SELECT 
                source_platform,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM f_offre), 2) as pct
            FROM f_offre
            GROUP BY source_platform
            ORDER BY count DESC
        """
    }
    
    for check_name, query in checks.items():
        print("\n{}".format(check_name))
        print("-" * 60)
        try:
            result = con.execute(query).fetchdf()
            print(result.to_string(index=False))
        except Exception as e:
            print("  ❌ Check failed: {}".format(e))

def run_analytics_queries(con: duckdb.DuckDBPyConnection) -> None:
    """Run analytical queries on the star schema"""
    print("\n" + "=" * 80)
    print("PHASE 12: ANALYTICAL QUERIES ON STAR SCHEMA")
    print("=" * 80)
    
    queries = {
        "Top 10 Companies": """
            SELECT company_name, COUNT(*) as nb_offres
            FROM f_offre
            WHERE company_name IS NOT NULL AND company_name != ''
            GROUP BY company_name
            ORDER BY nb_offres DESC
            LIMIT 10
        """,
        
        "Job Distribution by Region": """
            SELECT 
                h.nom_region,
                COUNT(f.job_id) as nb_offres
            FROM f_offre f
            LEFT JOIN d_localisation l ON f.id_ville = l.id_ville
            LEFT JOIN h_region h ON l.id_region = h.id_region
            WHERE h.nom_region IS NOT NULL AND h.nom_region != 'UNKNOWN'
            GROUP BY h.nom_region
            ORDER BY nb_offres DESC
        """,
        
        "Contract Type Distribution": """
            SELECT 
                c.type_contrat,
                COUNT(f.job_id) as nb_offres,
                ROUND(COUNT(f.job_id) * 100.0 / (SELECT COUNT(*) FROM f_offre), 2) as pct
            FROM f_offre f
            JOIN d_contrat c ON f.id_contrat = c.id_contrat
            WHERE c.type_contrat != 'UNKNOWN'
            GROUP BY c.type_contrat
            ORDER BY nb_offres DESC
        """,
    }
    
    for query_name, query in queries.items():
        print("\n{}".format(query_name))
        print("-" * 60)
        try:
            result = con.execute(query).fetchdf()
            print(result.to_string(index=False))
        except Exception as e:
            print("  ❌ Query failed: {}".format(e))

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Execute complete pipeline: ETL + Cleaning + Star Schema (IMPROVED)"""
    print("=" * 80)
    print("🚀 COMPLETE JOB MARKET ANALYSIS PIPELINE - IMPROVED VERSION")
    print("=" * 80)
    print("Configuration:")
    print("  MongoDB Database: {}".format(DATABASE_NAME))
    print("  Collections: {}".format(', '.join(COLLECTIONS)))
    print("  Document limit: {}".format(LIMIT if LIMIT else 'ALL'))
    print("  Similarity threshold: {}".format(SIMILARITY_THRESHOLD))
    print("  MotherDuck Database: {}".format(MOTHERDUCK_DATABASE))
    print("\n✅ Improvements:")
    print("  • Complete region mapping (13 regions)")
    print("  • UNKNOWN default values in all dimensions")
    print("  • Nullable foreign keys in fact table")
    print("  • Data quality validation")
    
    # PHASE 1: ETL
    try:
        df_raw = create_unified_dataset(limit=LIMIT)
        df_raw.to_excel(OUTPUT_RAW, index=False, engine='openpyxl')
        print("\n✅ Raw data saved: {} ({} records)".format(OUTPUT_RAW, len(df_raw)))
    except Exception as e:
        print("\n❌ ETL Pipeline failed: {}".format(e))
        import traceback
        traceback.print_exc()
        return None
    
    # PHASE 2: CLEANING
    try:
        df_cleaned = clean_job_data(df_raw, output_file=OUTPUT_CLEANED)
    except Exception as e:
        print("\n❌ Cleaning failed: {}".format(e))
        import traceback
        traceback.print_exc()
        return df_raw
    
    # PHASE 3: STAR SCHEMA (IMPROVED)
    con = None
    try:
        con = connect_motherduck()
        create_star_schema_ddl(con)
        
        # ✅ Utilisation des fonctions améliorées
        populate_dimension_tables(df_cleaned, con)
        records_inserted, validation_stats = populate_fact_table(df_cleaned, con)
        
        # ✅ Nouveau: Validation qualité
        run_data_quality_checks(con)
        
        # Analytics
        run_analytics_queries(con)
        
        print("\n✅ Star Schema successfully deployed to MotherDuck")
        
    except Exception as e:
        print("\n❌ Star Schema deployment failed: {}".format(e))
        import traceback
        traceback.print_exc()
    finally:
        if con:
            con.close()
            print("\n✅ MotherDuck connection closed")
    
    # FINAL SUMMARY
    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETE - SUMMARY")
    print("=" * 80)
    print("Files created:")
    print("  1. {} - {} records (raw)".format(OUTPUT_RAW, len(df_raw)))
    print("  2. {} - {} records (cleaned)".format(OUTPUT_CLEANED, len(df_cleaned)))
    print("\nData quality:")
    print("  Records removed: {}".format(len(df_raw) - len(df_cleaned)))
    print("  Retention rate: {:.1f}%".format(len(df_cleaned)/len(df_raw)*100))
    print("\nMotherDuck Star Schema:")
    print("  Database: {}".format(MOTHERDUCK_DATABASE))
    print("  Dimension tables: 4 (with UNKNOWN values)")
    print("  Fact table: 1 (with nullable FKs)")
    
    if con:
        try:
            count = con.execute("SELECT COUNT(*) FROM f_offre").fetchone()[0]
            retention = (count / len(df_cleaned)) * 100
            print("  ✅ Total records in f_offre: {} ({:.1f}% retention)".format(count, retention))
            
            if retention < 95:
                print("  ⚠️ WARNING: Data loss detected! Check validation logs.")
            else:
                print("  🎯 Excellent: Minimal data loss!")
        except:
            pass
    
    return df_cleaned


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    df_final = main()
    
    if df_final is not None:
        print("\n" + "=" * 80)
        print("✅ SUCCESS - Dataset and Star Schema ready for analysis")
        print("=" * 80)
        print("Variable 'df_final' contains {} cleaned job offers".format(len(df_final)))
        print("\nQuick preview:")
        print(df_final.head())
    else:
        print("\n" + "=" * 80)
        print("❌ PIPELINE FAILED")
        print("=" * 80)
        print("Check error messages above for details")