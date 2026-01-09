import streamlit as st
import duckdb
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import MOTHERDUCK_DATABASE
import numpy as np
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv()) 

# Vérification token
token = os.getenv('MOTHERDUCK_TOKEN')
if not token:
    st.error("ERREUR: Token MotherDuck manquant!")
    st.stop()

MOTHERDUCK_TOKEN = token

# Configuration page
st.set_page_config(layout="wide", page_title="Recherche Sémantique", page_icon="🔍")

# CSS moderne et épuré
st.markdown("""
<style>
    /* Import police moderne */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Reset */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Header personnalisé */
    .search-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.2);
    }
    
    .search-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    
    .search-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        text-align: center;
        margin: 0.5rem 0 0 0;
    }
    
    /* Input de recherche */
    .stTextInput > div > div > input {
        border: 2px solid #e9ecef;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
    }
    
    /* Bouton de recherche */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Metrics cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* Job cards */
    .job-card {
        background: white;
        border-radius: 16px;
        padding: 1.75rem;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .job-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border-color: #667eea;
    }
    
    .job-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1a202c;
        margin-bottom: 0.5rem;
        line-height: 1.3;
    }
    
    .job-company {
        font-size: 1rem;
        color: #718096;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    .job-description {
        color: #4a5568;
        font-size: 0.95rem;
        line-height: 1.7;
        margin: 1rem 0;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    /* Badges */
    .badge-container {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    
    .badge {
        display: inline-block;
        padding: 0.4rem 0.9rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .badge-location {
        background: #e6fffa;
        color: #234e52;
    }
    
    .badge-contract {
        background: #feebc8;
        color: #7c2d12;
    }
    
    .badge-score {
        background: #e9d8fd;
        color: #44337a;
    }
    
    /* Skills section */
    .skills-section {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #e9ecef;
    }
    
    .skills-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #718096;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .skills-text {
        font-size: 0.9rem;
        color: #4a5568;
        line-height: 1.5;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("# 🏠 Home Page")

try:
    cols = st.sidebar.columns(2)
    with cols[0]:
        st.image("./static/Logo3.png", width='content')
    with cols[1]:
        st.image("./static/Logo4_bis.png", width='content')
except:
    pass

# Header avec logos
try:
    logo_cols = st.columns(4)
    with logo_cols[0]:
        st.image("./static/Logo3.png", width='stretch')
    with logo_cols[1]:
        st.image("./static/Logo4_bis.png", width='stretch')
    with logo_cols[2]:
        st.image("./static/Logo_bis.png", width='stretch')
    with logo_cols[3]:
        st.image("./static/Logo2_bis.png", width='stretch')
except:
    pass

st.markdown("<br>", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="search-header">
    <h1>🔍 Recherche Sémantique d'Emploi</h1>
    <p>Trouvez votre poste idéal grâce à l'intelligence artificielle</p>
</div>
""", unsafe_allow_html=True)

# Cache resources
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

@st.cache_resource
def get_connection():
    connection_string = f"md:{MOTHERDUCK_DATABASE}?motherduck_token={MOTHERDUCK_TOKEN}"
    return duckdb.connect(connection_string)

def semantic_search(query_embedding: np.ndarray, top_k: int = 50, 
                    ville_filter: str = None, contrat_filter: str = None) -> pd.DataFrame:
    con = get_connection()
    embedding_list = query_embedding.tolist()
    
    where_clauses = ["f.embedding IS NOT NULL"]
    if ville_filter and ville_filter != "Toutes":
        where_clauses.append(f"l.ville = '{ville_filter}'")
    if contrat_filter and contrat_filter != "Tous":
        where_clauses.append(f"c.type_contrat = '{contrat_filter}'")
    
    where_sql = " AND ".join(where_clauses)
    
    query = f"""
    SELECT 
        f.job_id,
        f.title,
        f.company_name,
        COALESCE(l.ville, 'Non spécifié') AS ville,
        COALESCE(l.code_postal, '') AS code_postal,
        COALESCE(c.type_contrat, 'Non spécifié') AS type_contrat,
        f.description,
        COALESCE(f.hard_skills, '') AS hard_skills,
        array_cosine_similarity(f.embedding, ?::FLOAT[768]) AS similarity_score
    FROM f_offre f
    LEFT JOIN d_localisation l ON f.id_ville = l.id_ville
    LEFT JOIN d_contrat c ON f.id_contrat = c.id_contrat
    WHERE {where_sql}
    ORDER BY similarity_score DESC
    LIMIT {top_k}
    """
    
    try:
        df = con.execute(query, [embedding_list]).fetchdf()
        df['similarity_score'] = (df['similarity_score'] * 100).round(1)
        return df
    except Exception as e:
        st.error(f"Erreur recherche: {e}")
        return pd.DataFrame()

def get_filter_options():
    con = get_connection()
    try:
        villes = con.execute("""
            SELECT DISTINCT ville FROM d_localisation 
            WHERE ville IS NOT NULL ORDER BY ville
        """).fetchdf()['ville'].tolist()
        
        contrats = con.execute("""
            SELECT DISTINCT type_contrat FROM d_contrat 
            WHERE type_contrat IS NOT NULL ORDER BY type_contrat
        """).fetchdf()['type_contrat'].tolist()
        
        return villes, contrats
    except:
        return [], []

def render_job_card_native(job: dict):
    """Rendu de carte avec composants natifs Streamlit"""
    with st.container():
        st.markdown('<div class="job-card">', unsafe_allow_html=True)
        
        # Titre et entreprise
        st.markdown(f'<div class="job-title">{job.get("title", "Sans titre")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="job-company">{job.get("company_name", "Entreprise non spécifiée")}</div>', unsafe_allow_html=True)
        
        # Badges
        ville = job.get('ville', 'Non spécifié')
        code_postal = job.get('code_postal', '')
        location = f"{ville} ({code_postal})" if code_postal else ville
        contrat = job.get('type_contrat', 'Non spécifié')
        score = job.get('similarity_score', 0)
        
        st.markdown(f"""
        <div class="badge-container">
            <span class="badge badge-location">📍 {location}</span>
            <span class="badge badge-contract">📋 {contrat}</span>
            <span class="badge badge-score">⭐ {score}%</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Description
        description = job.get('description', '')
        if len(description) > 250:
            description = description[:250] + '...'
        st.markdown(f'<div class="job-description">{description}</div>', unsafe_allow_html=True)
        
        # Compétences
        hard_skills = job.get('hard_skills', '')
        if hard_skills:
            st.markdown(f"""
            <div class="skills-section">
                <div class="skills-label">Compétences requises</div>
                <div class="skills-text">{hard_skills}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Chargement modèle
with st.spinner('🤖 Chargement du modèle IA...'):
    model = load_model()

# Sidebar - Paramètres
with st.sidebar:
    st.markdown("### ⚙️ Paramètres de recherche")
    
    villes, contrats = get_filter_options()
    
    ville_filter = st.selectbox(
        "📍 Localisation",
        ["Toutes"] + villes,
        index=0
    )
    
    contrat_filter = st.selectbox(
        "📋 Type de contrat",
        ["Tous"] + contrats,
        index=0
    )
    
    top_k = st.slider(
        "📊 Nombre de résultats",
        min_value=10,
        max_value=100,
        value=50,
        step=10
    )
    
    st.markdown("---")
    st.markdown("### 💡 Conseils")
    st.info("""
    **Exemples de recherches:**
    
    ✅ Data Analyst CDI à Paris
    
    ✅ Stage développement web
    
    ✅ ML engineer expérimenté
    
    ✅ Alternance data science Lyon
    """)

# Zone de recherche
query = st.text_input(
    "",
    placeholder="Ex: Data Scientist avec Python et Machine Learning à Lyon...",
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search_button = st.button("🚀 Lancer la recherche", use_container_width=True)

# Exécution de la recherche
if search_button and query:
    with st.spinner('🔎 Recherche en cours...'):
        query_embedding = model.encode(query, convert_to_numpy=True)
        results = semantic_search(
            query_embedding=query_embedding,
            top_k=top_k,
            ville_filter=ville_filter if ville_filter != "Toutes" else None,
            contrat_filter=contrat_filter if contrat_filter != "Tous" else None
        )
    
    if len(results) > 0:
        # Statistiques
        st.markdown("<br>", unsafe_allow_html=True)
        metric_cols = st.columns(2)
        
        with metric_cols[0]:
            st.metric(
                label="📊 Offres trouvées",
                value=f"{len(results)}",
                delta=None
            )
        
        with metric_cols[1]:
            avg_score = results['similarity_score'].mean()
            st.metric(
                label="⭐ Pertinence moyenne",
                value=f"{avg_score:.1f}%",
                delta=None
            )
        
        st.markdown("---")
        st.markdown("### 🎯 Résultats")
        
        # Affichage des résultats en grille (2 colonnes)
        for i in range(0, len(results), 2):
            cols = st.columns(2)
            
            # Première carte
            with cols[0]:
                if i < len(results):
                    job = results.iloc[i].to_dict()
                    render_job_card_native(job)
            
            # Deuxième carte
            with cols[1]:
                if i + 1 < len(results):
                    job = results.iloc[i + 1].to_dict()
                    render_job_card_native(job)
            
            # Espacement entre les lignes
            if i + 2 < len(results):
                st.markdown("<br>", unsafe_allow_html=True)
    
    else:
        st.warning("⚠️ Aucun résultat trouvé. Essayez avec des termes différents ou ajustez les filtres.")

elif search_button and not query:
    st.error("❌ Veuillez entrer une requête de recherche.")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown(
    "<div style='text-align: center; color: #718096; font-size: 0.9rem;'>"
    "Powered by <strong>MotherDuck</strong> × <strong>Sentence Transformers</strong> | "
    "RUCHE Team © 2026"
    "</div>",
    unsafe_allow_html=True
)
