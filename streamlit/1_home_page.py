import streamlit as st
import duckdb
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import MOTHERDUCK_DATABASE
import numpy as np
import html
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

token = os.getenv("MOTHERDUCK_TOKEN")
if not token:
    st.error("ERREUR: Token MotherDuck manquant!")
    st.stop()

MOTHERDUCK_TOKEN = token
st.set_page_config(layout="wide", page_title="Recherche Sémantique", page_icon="🔍")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%); }
    .search-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem; border-radius: 16px; margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.2);
    }
    .search-header h1 { color: white; font-size: 2.5rem; font-weight: 700; margin: 0; text-align: center; }
    .search-header p { color: rgba(255, 255, 255, 0.9); font-size: 1.1rem; text-align: center; margin: 0.5rem 0 0 0; }
    
    /* Search bar enhanced */
    .search-container {
        position: relative;
        max-width: 900px;
        margin: 2rem auto;
    }
    .stTextInput > div > div > input {
        border: 3px solid #667eea !important;
        border-radius: 50px !important;
        padding: 1.2rem 3rem 1.2rem 4rem !important;
        font-size: 1.15rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.15) !important;
        background: white !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #764ba2 !important;
        box-shadow: 0 12px 32px rgba(102, 126, 234, 0.25) !important;
        transform: translateY(-2px);
    }
    .stTextInput > div > div > input::placeholder {
        color: #a0aec0;
        font-weight: 400;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 50px; padding: 1.2rem 3rem;
        font-size: 1.15rem; font-weight: 600; transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    .stButton > button:hover { 
        transform: translateY(-3px); 
        box-shadow: 0 10px 28px rgba(102, 126, 234, 0.5);
    }
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; color: #667eea; }
    .job-card {
        background: white; border-radius: 16px; padding: 1.75rem; border: 1px solid #e9ecef;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); transition: all 0.3s ease; height: 100%;
    }
    .job-card:hover { transform: translateY(-4px); box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); border-color: #667eea; }
    .job-title { font-size: 1.35rem; font-weight: 700; color: #1a202c; margin-bottom: 0.5rem; line-height: 1.3; }
    .job-title a { color: #1a202c; text-decoration: none; }
    .job-title a:hover { color: #667eea; text-decoration: underline; }
    .job-company { font-size: 1rem; color: #718096; font-weight: 500; margin-bottom: 1rem; }
    .badge-container { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 1rem 0; }
    .badge { display: inline-block; padding: 0.4rem 0.9rem; border-radius: 8px; font-size: 0.85rem; font-weight: 600; }
    .badge-location { background: #e6fffa; color: #234e52; }
    .badge-contract { background: #feebc8; color: #7c2d12; }
    .badge-score { background: #e9d8fd; color: #44337a; }
    .skills-section { margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e9ecef; }
    .skills-label { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #718096; letter-spacing: 0.5px; margin-bottom: 0.5rem; }
    .skills-text { font-size: 0.9rem; color: #4a5568; line-height: 1.5; }
    hr { margin: 2rem 0; border: none; border-top: 1px solid #e9ecef; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'><b></b></h1>", unsafe_allow_html=True)
st.sidebar.image("./static/Logo3.png", width=150)
st.sidebar.markdown("# Home page")

with st.container(horizontal=True):
    col1, col2 = st.columns(2)
    with col1:
        left = st.container(horizontal_alignment="center", border=False)
        left.markdown("<h3 style='text-align: center;'>Une application de recherche d'emploi data/ia <br/>et d'analyse du marché</h3>", unsafe_allow_html=True)
        left.image("./static/Logo3.png", width=210, caption="")
    with col2:
        right = st.container(horizontal_alignment="center", border=False)
        right.markdown("<h3 style='text-align: center;'>Architecture</h3>", unsafe_allow_html=True)
        right.image("./static/architecture.png", width="stretch", caption="")

st.divider()
st.markdown("""
<div class="search-header">
    <h1>Trouvez le job de vos rêves dans le domaine de la data et de l'intelligence artificielle</h1>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

@st.cache_resource
def get_connection():
    return duckdb.connect(f"md:{MOTHERDUCK_DATABASE}?motherduck_token={MOTHERDUCK_TOKEN}")

@st.cache_data(ttl=3600)
def get_total_offers():
    con = get_connection()
    return con.execute("SELECT COUNT(*) FROM f_offre WHERE embedding IS NOT NULL").fetchone()[0]

def semantic_search(query_embedding: np.ndarray, top_k: int = 50, 
                    ville_filter: list = None, contrat_filter: list = None,
                    min_similarity: float = 0.0) -> pd.DataFrame:
    con = get_connection()
    embedding_list = query_embedding.tolist()
    
    where_clauses = ["f.embedding IS NOT NULL"]
    if ville_filter and len(ville_filter) > 0:
        villes_sql = "', '".join(ville_filter)
        where_clauses.append(f"l.ville IN ('{villes_sql}')")
    if contrat_filter and len(contrat_filter) > 0:
        contrats_sql = "', '".join(contrat_filter)
        where_clauses.append(f"c.type_contrat IN ('{contrats_sql}')")
    
    where_sql = " AND ".join(where_clauses)
    
    query = f"""
    WITH scored AS (
        SELECT 
            f.job_id, f.title, f.company_name, f.description, f.source_url,
            COALESCE(l.ville, 'Non spécifié') AS ville,
            COALESCE(l.code_postal, '') AS code_postal,
            COALESCE(c.type_contrat, 'Non spécifié') AS type_contrat,
            COALESCE(f.hard_skills, '') AS hard_skills,
            ROUND(array_cosine_similarity(f.embedding, ?::FLOAT[768]) * 100, 1) AS similarity_score
        FROM f_offre f
        LEFT JOIN d_localisation l ON f.id_ville = l.id_ville
        LEFT JOIN d_contrat c ON f.id_contrat = c.id_contrat
        WHERE {where_sql}
    )
    SELECT * FROM scored 
    WHERE similarity_score >= {min_similarity}
    ORDER BY similarity_score DESC
    LIMIT {top_k}
    """
    
    try:
        return con.execute(query, [embedding_list]).fetchdf()
    except Exception as e:
        st.error(f"Erreur recherche: {e}")
        return pd.DataFrame()

def get_filter_options():
    con = get_connection()
    try:
        villes = con.execute("SELECT DISTINCT ville FROM d_localisation WHERE ville IS NOT NULL ORDER BY ville").fetchdf()['ville'].tolist()
        contrats = con.execute("SELECT DISTINCT type_contrat FROM d_contrat WHERE type_contrat IS NOT NULL ORDER BY type_contrat").fetchdf()['type_contrat'].tolist()
        return villes, contrats
    except:
        return [], []

def render_job_card_native(job: dict):
    with st.container():
        st.markdown('<div class="job-card">', unsafe_allow_html=True)
        
        title = html.escape(job.get("title") or "Sans titre")
        source_url = job.get("source_url") or ""
        
        if source_url:
            st.markdown(f'<div class="job-title"><a href="{source_url}" target="_blank">{title}</a></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="job-title">{title}</div>', unsafe_allow_html=True)
        
        company = html.escape(job.get("company_name") or "Entreprise non spécifiée")
        st.markdown(f'<div class="job-company">{company}</div>', unsafe_allow_html=True)
        
        ville = html.escape(job.get("ville") or "Non spécifié")
        code_postal = html.escape(job.get("code_postal") or "")
        location = f"{ville} ({code_postal})" if code_postal else ville
        contrat = html.escape(job.get("type_contrat") or "Non spécifié")
        score = job.get("similarity_score", 0)
        
        st.markdown(f"""
        <div class="badge-container">
            <span class="badge badge-location">📍 {location}</span>
            <span class="badge badge-contract">📋 {contrat}</span>
            <span class="badge badge-score">⭐ {score:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
        
        description = job.get("description") or ""
        if description:
            preview = description[:200] + "..." if len(description) > 200 else description
            preview_safe = html.escape(preview).replace('\n', '<br>')
            
            st.markdown(f'<div style="color: #4a5568; font-size: 0.95rem; margin: 1rem 0; line-height: 1.6;">{preview_safe}</div>', unsafe_allow_html=True)
            
            if len(description) > 200:
                with st.expander("📖 Lire la suite"):
                    st.text(description)
        
        hard_skills = job.get("hard_skills") or ""
        if hard_skills:
            skills_safe = html.escape(hard_skills)
            st.markdown(f"""
            <div class="skills-section">
                <div class="skills-label">Compétences requises</div>
                <div class="skills-text">{skills_safe}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

with st.spinner("🤖 Chargement du modèle IA..."):
    model = load_model()

with st.sidebar:
    st.markdown("### ⚙️ Paramètres de recherche")
    villes, contrats = get_filter_options()
    
    ville_filter = st.multiselect("📍 Localisation", villes, default=[])
    contrat_filter = st.multiselect("📋 Type de contrat", contrats, default=[])
    
    min_similarity = st.slider(
        "🎯 Score minimum de pertinence (%)",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        help="Filtrer les offres ayant un score de similarité supérieur ou égal à ce seuil"
    )
    
    top_k = st.slider("📊 Nombre de résultats", 10, 100, 50, 10)
    
    st.markdown("---")
    st.markdown("### 💡 Conseils de recherche")
    st.info("**Exemples de requêtes :**\n\n⚙ Data Analyst CDI Paris\n\n⚙ Stage Java/Spark\n\n⚙ ML engineer expérimenté\n\n⚙ Alternance data science Lyon")

st.markdown('<div class="search-container">', unsafe_allow_html=True)
query = st.text_input(
    "",
    placeholder="🔎  Ex: Data Scientist Python et Machine Learning à Lyon...",
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    search_button = st.button("🚀 Lancer la recherche", use_container_width=True)

if search_button and query:
    with st.spinner("🔎 Recherche en cours..."):
        query_embedding = model.encode(query, convert_to_numpy=True)
        results = semantic_search(
            query_embedding, 
            top_k, 
            ville_filter if ville_filter else None, 
            contrat_filter if contrat_filter else None,
            min_similarity
        )
    
    if len(results) > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        metric_cols = st.columns(2)
        
        total_offers = get_total_offers()
        
        with metric_cols[0]:
            st.metric("📊 Base de données", f"{total_offers:,} offres")
        with metric_cols[1]:
            st.metric("⭐ Pertinence moyenne", f"{results['similarity_score'].mean():.1f}%")
        
        st.markdown("---")
        st.markdown(f"### 🎯 {len(results)} résultats trouvés")
        
        for i in range(0, len(results), 2):
            cols = st.columns(2)
            with cols[0]:
                if i < len(results):
                    render_job_card_native(results.iloc[i].to_dict())
            with cols[1]:
                if i + 1 < len(results):
                    render_job_card_native(results.iloc[i + 1].to_dict())
            if i + 2 < len(results):
                st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Aucun résultat trouvé. Essayez d'abaisser le score minimum de pertinence.")
elif search_button:
    st.error("❌ Veuillez entrer une requête.")

st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown("<div style='text-align: center; color: #718096; font-size: 0.9rem;'>Powered by <strong>MotherDuck</strong> × <strong>Sentence Transformers</strong> | RUCHE Team © 2026</div>", unsafe_allow_html=True)