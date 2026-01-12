import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from ruche.db import get_connection
import numpy as np
import html
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path


load_dotenv(find_dotenv())

st.set_page_config(layout="wide", page_title="Recherche Sémantique", page_icon="🔍")

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%); }
    
    /* Header épuré */
    .search-header {
        background: #f5f7fa;
        padding: 4rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .search-header h1 { 
        color: #5a67d8;
        font-size: 3rem;
        font-weight: 600;
        margin: 0 0 1rem 0;
        letter-spacing: -0.5px;
    }
    .search-header p {
        color: #4a5568;
        font-size: 1.1rem;
        margin: 0;
        font-weight: 400;
        max-width: 700px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Search bar avec bordure dorée */
    .search-container {
        max-width: 750px;
        margin: 2rem auto 1.5rem auto;
    }
    .stTextInput > div > div > input {
        border: 3px solid #f6ad55 !important;
        border-radius: 50px !important;
        padding: 1.3rem 2rem !important;
        font-size: 1.05rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(246, 173, 85, 0.2) !important;
        background: white !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #ed8936 !important;
        box-shadow: 0 6px 16px rgba(246, 173, 85, 0.3) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #cbd5e0;
        font-weight: 400;
    }
    
    /* Bouton bleu */
    .stButton > button {
        background: #4c51bf;
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 2.5rem;
        font-size: 1.05rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(76, 81, 191, 0.4);
    }
    .stButton > button:hover { 
        background: #5a67d8;
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(76, 81, 191, 0.5);
    }
    
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; color: #5a67d8; }
    .job-card {
        background: white; border-radius: 16px; padding: 1.75rem; border: 1px solid #e9ecef;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); transition: all 0.3s ease; height: 100%;
    }
    .job-card:hover { transform: translateY(-4px); box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); border-color: #5a67d8; }
    .job-title { font-size: 1.35rem; font-weight: 700; color: #1a202c; margin-bottom: 0.5rem; line-height: 1.3; }
    .job-title a { color: #1a202c; text-decoration: none; }
    .job-title a:hover { color: #5a67d8; text-decoration: underline; }
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
""",
    unsafe_allow_html=True,
)
CURRENT_DIR = Path(__file__).resolve().parent
LOGO_PATH = CURRENT_DIR / "static" / "Logo3.png"

st.sidebar.image(str(LOGO_PATH), width=150)
st.sidebar.markdown("# Filtres")

st.markdown(
    """
<div class="search-header">
    <h1>Trouvez votre futur job Data & IA</h1>
    <p>Le moteur de recherche sémantique intelligent qui comprend votre langage,<br>pas juste vos mots-clés.</p>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )


@st.cache_data(ttl=3600)
def get_total_offers():
    con = get_connection()
    return con.execute(
        "SELECT COUNT(*) FROM f_offre WHERE embedding IS NOT NULL"
    ).fetchone()[0]


@st.cache_data(ttl=3600)
def get_regions():
    """Récupère les régions distinctes (sans UNKNOWN)"""
    con = get_connection()
    try:
        regions = (
            con.execute(
                """
            SELECT DISTINCT nom_region 
            FROM h_region 
            WHERE nom_region IS NOT NULL 
                AND nom_region != 'UNKNOWN'
            ORDER BY nom_region
        """
            )
            .fetchdf()["nom_region"]
            .tolist()
        )
        return regions
    except Exception as e:
        st.error(f"Erreur chargement régions: {e}")
        return []


def semantic_search(
    query_embedding: np.ndarray,
    top_k: int = 50,
    region_filter: list = None,
    contract_filters: dict = None,
    min_similarity: float = 0.0,
) -> pd.DataFrame:
    con = get_connection()
    embedding_list = query_embedding.tolist()

    where_clauses = ["f.embedding IS NOT NULL", "f.is_duplicate = FALSE"]

    # Filtre par région
    if region_filter and len(region_filter) > 0:
        regions_sql = "', '".join([r.replace("'", "''") for r in region_filter])
        where_clauses.append(f"r.nom_region IN ('{regions_sql}')")

    # Filtre par contrat (checkboxes)
    if contract_filters:
        selected_contracts = [k for k, v in contract_filters.items() if v]
        if selected_contracts:
            contracts_sql = "', '".join(
                [c.replace("'", "''") for c in selected_contracts]
            )
            where_clauses.append(f"c.type_contrat IN ('{contracts_sql}')")

    where_sql = " AND ".join(where_clauses)

    query = f"""
    WITH scored AS (
        SELECT 
            f.job_id, f.title, f.company_name, f.description, f.source_url,
            COALESCE(l.ville, 'Non spécifié') AS ville,
            COALESCE(l.code_postal, '') AS code_postal,
            COALESCE(r.nom_region, 'Non spécifié') AS region,
            COALESCE(c.type_contrat, 'Non spécifié') AS type_contrat,
            COALESCE(f.hard_skills, '') AS hard_skills,
            ROUND(array_cosine_similarity(f.embedding, ?::FLOAT[768]) * 100, 1) AS similarity_score
        FROM f_offre f
        LEFT JOIN d_localisation l ON f.id_ville = l.id_ville
        LEFT JOIN h_region r ON l.id_region = r.id_region
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


def render_job_card_native(job: dict):
    with st.container():
        st.markdown('<div class="job-card">', unsafe_allow_html=True)

        title = html.escape(job.get("title") or "Sans titre")
        source_url = job.get("source_url") or ""

        if source_url:
            st.markdown(
                f'<div class="job-title"><a href="{source_url}" target="_blank">{title}</a></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f'<div class="job-title">{title}</div>', unsafe_allow_html=True)

        company = html.escape(job.get("company_name") or "Entreprise non spécifiée")
        st.markdown(f'<div class="job-company">{company}</div>', unsafe_allow_html=True)

        ville = html.escape(job.get("ville") or "Non spécifié")
        region = html.escape(job.get("region") or "")
        code_postal = html.escape(job.get("code_postal") or "")

        location_parts = []
        if ville:
            location_parts.append(f"{ville} ({code_postal})" if code_postal else ville)
        if region:
            location_parts.append(region)
        location = " • ".join(location_parts) if location_parts else "Non spécifié"

        contrat = html.escape(job.get("type_contrat") or "Non spécifié")
        score = job.get("similarity_score", 0)

        st.markdown(
            f"""
        <div class="badge-container">
            <span class="badge badge-location">📍 {location}</span>
            <span class="badge badge-contract">📋 {contrat}</span>
            <span class="badge badge-score">⭐ {score:.1f}%</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        description = job.get("description") or ""
        if description:
            preview = (
                description[:200] + "..." if len(description) > 200 else description
            )
            preview_safe = html.escape(preview).replace("\n", "<br>")

            st.markdown(
                f'<div style="color: #4a5568; font-size: 0.95rem; margin: 1rem 0; line-height: 1.6;">{preview_safe}</div>',
                unsafe_allow_html=True,
            )

            if len(description) > 200:
                with st.expander("📖 Lire la suite"):
                    st.text(description)

        hard_skills = job.get("hard_skills") or ""
        if hard_skills:
            skills_safe = html.escape(hard_skills)
            st.markdown(
                f"""
            <div class="skills-section">
                <div class="skills-label">Compétences requises</div>
                <div class="skills-text">{skills_safe}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


with st.spinner("🤖 Chargement du modèle IA..."):
    model = load_model()

with st.sidebar:
    # st.markdown("### ⚙️ Paramètres de recherche")

    # Filtre Région (multiselect)
    regions = get_regions()
    st.markdown("#### Région")
    region_filter = st.multiselect(
        "Sélectionnez une ou plusieurs régions",
        options=regions,
        default=[],
        label_visibility="collapsed",
    )

    # Filtre Type de contrat (checkboxes)
    st.markdown("#### Type de contrat")
    filter_cdi = st.checkbox("CDI", value=False)
    filter_cdd = st.checkbox("CDD", value=False)
    filter_stage = st.checkbox("Stage", value=False)
    filter_alternance = st.checkbox("Alternance / Apprentissage", value=False)
    filter_freelance = st.checkbox("Freelance", value=False)
    filter_interim = st.checkbox("Intérim", value=False)
    filter_public = st.checkbox("Contrat public", value=False)

    st.markdown("---")

    min_similarity = st.slider(
        "🎯 Score minimum de pertinence (%)",
        min_value=0,
        max_value=100,
        value=70,
        step=5,
        help="Filtrer les offres ayant un score de similarité supérieur ou égal à ce seuil",
    )

    top_k = st.slider("📊 Nombre de résultats", 10, 100, 50, 10)

    st.markdown("---")
    st.markdown("### 💡 Conseils de recherche textuelle")
    st.info(
        "⚙ Data Analyst CDI Île-de-France\n\n⚙ Stage Java/Spark\n\n⚙ ML engineer expérimenté\n\n⚙ Alternance data science Auvergne-Rhône-Alpes"
    )

st.markdown('<div class="search-container">', unsafe_allow_html=True)
query = st.text_input(
    "",
    placeholder="Ex: Data Scientist Junior NLP à Lyon...",
    label_visibility="collapsed",
)
st.markdown("</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    search_button = st.button("Lancer la recherche 🚀", use_container_width=True)

if search_button and query:
    # Préparer les filtres contrat
    contract_filters = {
        "CDI": filter_cdi,
        "CDD": filter_cdd,
        "STAGE": filter_stage,
        "ALTERNANCE": filter_alternance,
        "INTERIM": filter_interim,
        "AUTRE": filter_freelance,
        "CONTRAT_PUBLIC": filter_public,
    }

    with st.spinner("🔎 Recherche en cours..."):
        query_embedding = model.encode(query, convert_to_numpy=True)
        results = semantic_search(
            query_embedding,
            top_k,
            region_filter if region_filter else None,
            contract_filters if any(contract_filters.values()) else None,
            min_similarity,
        )

    if len(results) > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        metric_cols = st.columns(2)

        total_offers = get_total_offers()

        with metric_cols[0]:
            st.metric("📊 Base de données", f"{total_offers:,} offres")
        with metric_cols[1]:
            st.metric(
                "⭐ Pertinence moyenne", f"{results['similarity_score'].mean():.1f}%"
            )

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
        st.warning(
            "⚠️ Aucun résultat trouvé. Essayez d'abaisser le score minimum de pertinence."
        )
elif search_button:
    st.error("❌ Veuillez entrer une requête.")

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: #718096; font-size: 0.9rem;'>Powered by <strong>MotherDuck</strong> × <strong>Sentence Transformers</strong> | RUCHE Team © 2026</div>",
    unsafe_allow_html=True,
)
