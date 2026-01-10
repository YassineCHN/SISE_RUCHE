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

st.sidebar.markdown("## 🔍 Filtres")

limit = st.slider(
    "Nombre d'offres analysées",
    min_value=200,
    max_value=5000,
    step=200,
    value=1000
)

# Filtre contrat
st.sidebar.markdown("### 📋 Type de contrat")
filter_cdi = st.sidebar.checkbox("CDI", value=False)
filter_cdd = st.sidebar.checkbox("CDD", value=False)
filter_stage = st.sidebar.checkbox("Stage", value=False)
filter_alternance = st.sidebar.checkbox("Alternance / Apprentissage", value=False)
filter_freelance = st.sidebar.checkbox("Freelance", value=False)
filter_interim = st.sidebar.checkbox("Intérim", value=False)



# Filtre date
st.sidebar.markdown("### 📅 Date de publication")
date_filter = st.sidebar.radio(
    "Publié depuis",
    options=['Toutes', '7 jours', '21 jours', '1 mois', '3 mois'],
    index=0
)


# Filtre région
st.sidebar.markdown("### 📍 Région")
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
df = load_data(contract_filters, date_filter, region_filter, limit)

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
