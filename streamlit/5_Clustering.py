import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import plotly.express as px
import umap
import hdbscan
from sentence_transformers import SentenceTransformer
from streamlit.config import MOTHERDUCK_DATABASE,EMBEDDING_MODEL,UMAP_PARAMS, HDBSCAN_PARAMS,CONTRACT_FLAGS
from collections import Counter


con = duckdb.connect("MOTHERDUCK_DATABASE")
# ------------------------
# CACHE MODELE
# ------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)

# ------------------------
# TITRE DE PAGE 
# ------------------------
st.title(" Clustering des offres")

# ----------------------
# SIDEBAR : FILTRES
# ----------------------

st.sidebar.markdown("## Filtres")

limit = st.sidebar.slider(
    "Nombre d'offres analysées",
    min_value=200,
    max_value=5000,
    step=200,
    value=1000
)

# Filtre contrat
st.sidebar.markdown("###  Type de contrat")
contract_filter = st.sidebar.multiselect(
    "Sélectionner un ou plusieurs types de contrat",
    options=['Tous','CDI', 'CDD', 'CONTRAT_PUBLIC', 'INTERIM', 'ALTERNANCE', 'STAGE', 'AUTRE'],
    default=['Tous']
)



# Filtre date
st.sidebar.markdown("###  Date de publication")
date_filter = st.sidebar.radio(
    "Publié depuis",
    options=['Toutes', '7 jours', '21 jours', '1 mois', '3 mois'],
    index=0
)


# Filtre région
st.sidebar.markdown("###  Région")
region_filter = st.sidebar.multiselect(
    "Sélectionner une ou plusieurs régions",
    options=['Toutes','Auvergne-Rhône-Alpes', 'Bourgogne-Franche-Comté', 'Bretagne', 'Centre-Val de Loire', 'Corse', 'Grand Est', 'Hauts-de-France', 'Île-de-France', 'Normandie', 'Nouvelle-Aquitaine', 'Occitanie', 'Pays de la Loire', 'Provence-Alpes-Côte d\'Azur'],
    default=['Toutes']
)

# Bouton reset
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Réinitialiser les filtres", use_container_width=True):
    st.rerun()
    
# ------------------------
# DATA LOADING
# ------------------------
@st.cache_data

def load_data(con, limit, contract_filters=None, date_filter='Toutes', region_filter='Toutes'):
    con = duckdb.connect(MOTHERDUCK_DATABASE)
    
    """Charge les données avec filtres appliqués"""
    query = f"""
        SELECT
            job_id,
            title,
            description,
            hard_skills,
            soft_skills
        FROM f_offre f
        LEFT JOIN d_localisation l ON f.id_ville = l.id_ville
        LEFT JOIN h_region r ON l.id_region = r.id_region
        LEFT JOIN d_contrat c ON f.id_contrat = c.id_contrat
        LEFT JOIN d_date d ON f.id_date_publication = d.id_date
        WHERE description IS NOT NULL
        LIMIT {limit}
        
        # Filtre contrat
    if contract_filter and 'Tous' not in contract_filter:
        query += "\n    AND c.type_contrat IN ('" + "', '".join(contract_filter) + "')"
        
         # Filtre date
    if date_filter == '7 jours':
        query += "\n    AND d.date_complete >= CURRENT_DATE - INTERVAL '7 days'"
    elif date_filter == '21 jours':
        query += "\n    AND d.date_complete >= CURRENT_DATE - INTERVAL '21 days'"
    elif date_filter == '1 mois':
        query += "\n    AND d.date_complete >= CURRENT_DATE - INTERVAL '30 days'"
    elif date_filter == '3 mois':
        query += "\n    AND d.date_complete >= CURRENT_DATE - INTERVAL '90 days'"
        
         # Filtre région
    if region_filter and 'Toutes' not in region_filter:
        query += "\n    AND r.region_name IN ('" + "', '".join(region_filter) + "')"
    """
    df = con.execute(query).df()
    con.close()
    return df

# Préparer les filtres
contract_filters = {
    'cdi': filter_cdi,
    'cdd': filter_cdd,
    'stage': filter_stage,
    'alternance': filter_alternance,
    'freelance': filter_freelance,
    'interim': filter_interim
}

# ------------------------
# FEATURE ENGINEERING
# ------------------------
def build_ml_text(row):
    parts = [
        row["description"],
        row["title"] * 2 if row["title"] else "",
        " ".join(row["hard_skills"]) * 3 if row["hard_skills"] else "",
        " ".join(row["soft_skills"]) if row["soft_skills"] else "",
    ]
    return " ".join([p for p in parts if p])

# ------------------------
# CLUSTER LABELING
# ------------------------
def label_clusters(df, cluster_col="cluster_id", top_n=3):
    global_counter = Counter(
        skill for skills in df["hard_skills"] if skills for skill in skills
    )

    labels = {}
    for cid, group in df.groupby(cluster_col):
        if cid == -1:
            labels[cid] = "Offres atypiques"
            continue

        cluster_counter = Counter(
            skill for skills in group["hard_skills"] if skills for skill in skills
        )

        scores = {
            s: f / global_counter.get(s, 1)
            for s, f in cluster_counter.items()
        }

        top = sorted(scores, key=scores.get, reverse=True)[:top_n]
        labels[cid] = " / ".join(top)

    return labels

# -----------------------------------
# CHARGEMENT DES DONNEES ET CALCULS
# -----------------------------------
df = load_data(limit, contract_filters=contract_filters if any(contract_filters.values()) else None, date_filter=date_filter, region_filter=region_filter)

st.caption(f"📄 {len(df)} offres analysées après filtres")
df["ml_text"] = df.apply(build_ml_text, axis=1)

model = load_model()

with st.spinner("Calcul des embeddings…"):
    embeddings = model.encode(
        df["ml_text"].tolist(),
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=False
    )

with st.spinner("Clustering en cours…"):
    reducer = umap.UMAP(UMAP_PARAMS)
    coords = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(HDBSCAN_PARAMS)
    df["cluster_id"] = clusterer.fit_predict(embeddings)

df["x"], df["y"], df["z"] = coords[:, 0], coords[:, 1], coords[:, 2]
df["cluster_label"] = df["cluster_id"].map(label_clusters(df))

# ------------------------
# VISUALISATION
# ------------------------

fig = px.scatter_3d(
    filtered,
    x="umap_x",
    y="umap_y",
    z="umap_z",
    color="cluster_label",
    hover_data=["title", "hard_skills"],
    opacity=0.8,
)

st.plotly_chart(fig, use_container_width=True)
