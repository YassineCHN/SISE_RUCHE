import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import umap
import hdbscan
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, UMAP_PARAMS, HDBSCAN_PARAMS
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv
from ruche.db import get_connection

# ------------------------
# CONNECTION DB
# ------------------------
con = get_connection()

st.set_page_config(layout="wide", page_title="Clusters d'offres", page_icon="👨‍👧‍👧")


# ------------------------
# CACHE MODELE
# ------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# ------------------------
# TITRE DE PAGE
# ------------------------
st.title("🔬 Clustering des offres")
st.markdown(
    "Exploration de l'espace sémantique des offres d'emploi via réduction dimensionnelle UMAP et clustering HDBSCAN"
)
st.markdown("---")

# ----------------------
# SIDEBAR : FILTRES
# ----------------------
CURRENT_DIR = Path(__file__).resolve().parent
LOGO_PATH = CURRENT_DIR / "static" / "Logo3.png"
with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 🐝 RUCHE")
        st.image(str(LOGO_PATH), width=140)
st.sidebar.markdown("## 🔍 Filtres")

limit = st.sidebar.slider(
    "Nombre d'offres analysées", min_value=200, max_value=5000, step=200, value=1000
)

# Paramètres UMAP
st.sidebar.markdown("### 🎛️ Paramètres UMAP")
n_neighbors = st.sidebar.slider("n_neighbors", 5, 50, 15)
min_dist = st.sidebar.slider("min_dist", 0.0, 1.0, 0.1, 0.05)

# Paramètres HDBSCAN
st.sidebar.markdown("### 🎛️ Paramètres HDBSCAN")
min_cluster_size = st.sidebar.slider("min_cluster_size", 5, 100, 50, 5)
min_samples = st.sidebar.slider("min_samples", 1, 50, 10)

st.sidebar.markdown("---")

# Filtre contrat
st.sidebar.markdown("### 📋 Type de contrat")
contract_filter = st.sidebar.multiselect(
    "Sélectionner un ou plusieurs types de contrat",
    options=[
        "Tous",
        "CDI",
        "CDD",
        "CONTRAT_PUBLIC",
        "INTERIM",
        "ALTERNANCE",
        "STAGE",
        "AUTRE",
    ],
    default=["Tous"],
)

# Filtre date
st.sidebar.markdown("### 📅 Date de publication")
date_filter = st.sidebar.radio(
    "Publié depuis",
    options=["Toutes", "7 jours", "21 jours", "1 mois", "3 mois"],
    index=0,
)

# Filtre région
st.sidebar.markdown("### 🗺️ Région")
region_filter = st.sidebar.multiselect(
    "Sélectionner une ou plusieurs régions",
    options=[
        "Toutes",
        "Auvergne-Rhône-Alpes",
        "Bourgogne-Franche-Comté",
        "Bretagne",
        "Centre-Val de Loire",
        "Corse",
        "Grand Est",
        "Hauts-de-France",
        "Île-de-France",
        "Normandie",
        "Nouvelle-Aquitaine",
        "Occitanie",
        "Pays de la Loire",
        "Provence-Alpes-Côte d'Azur",
    ],
    default=["Toutes"],
)

# Bouton reset
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Réinitialiser les filtres", use_container_width=True):
    st.rerun()


# ------------------------
# DATA LOADING
# ------------------------
@st.cache_data
def load_data(
    _con, limit, contract_filter="Tous", date_filter="Toutes", region_filter="Toutes"
):
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
    """

    # Filtre contrat
    if contract_filter and "Tous" not in contract_filter:
        query += (
            "\n        AND c.type_contrat IN ('" + "', '".join(contract_filter) + "')"
        )

    # Filtre date
    if date_filter == "7 jours":
        query += "\n        AND d.date_complete >= CURRENT_DATE - INTERVAL '7 days'"
    elif date_filter == "21 jours":
        query += "\n        AND d.date_complete >= CURRENT_DATE - INTERVAL '21 days'"
    elif date_filter == "1 mois":
        query += "\n        AND d.date_complete >= CURRENT_DATE - INTERVAL '30 days'"
    elif date_filter == "3 mois":
        query += "\n        AND d.date_complete >= CURRENT_DATE - INTERVAL '90 days'"

    # Filtre région
    if region_filter and "Toutes" not in region_filter:
        query += "\n        AND r.nom_region IN ('" + "', '".join(region_filter) + "')"

    query += f"\n        LIMIT {limit}"

    df = _con.execute(query).df()
    return df


# ------------------------
# FEATURE ENGINEERING
# ------------------------
def build_ml_text(row):
    """Construit le texte enrichi pour l'embedding"""
    parts = []

    if pd.notna(row["description"]):
        parts.append(str(row["description"]))

    if pd.notna(row["title"]):
        parts.append(str(row["title"]) * 2)

    if pd.notna(row["hard_skills"]) and row["hard_skills"]:
        if isinstance(row["hard_skills"], str):
            skills_str = row["hard_skills"]
        elif isinstance(row["hard_skills"], list):
            skills_str = " ".join(row["hard_skills"])
        else:
            skills_str = str(row["hard_skills"])
        parts.append(skills_str * 3)

    if pd.notna(row["soft_skills"]) and row["soft_skills"]:
        if isinstance(row["soft_skills"], str):
            skills_str = row["soft_skills"]
        elif isinstance(row["soft_skills"], list):
            skills_str = " ".join(row["soft_skills"])
        else:
            skills_str = str(row["soft_skills"])
        parts.append(skills_str)

    return " ".join(parts)


# ------------------------
# CLUSTER LABELING
# ------------------------
def label_clusters(df, cluster_col="cluster_id", top_n=3):
    """Génère des labels pour chaque cluster basés sur les compétences surreprésentées"""

    # Compteur global de toutes les compétences
    global_counter = Counter()
    for skills in df["hard_skills"]:
        if pd.notna(skills) and skills:
            if isinstance(skills, str):
                skills_list = [s.strip() for s in skills.split(",") if s.strip()]
            elif isinstance(skills, list):
                skills_list = skills
            else:
                continue
            global_counter.update(skills_list)

    labels = {}
    for cid in df[cluster_col].unique():
        if cid == -1:
            labels[cid] = "Offres atypiques"
            continue

        group = df[df[cluster_col] == cid]
        cluster_counter = Counter()

        for skills in group["hard_skills"]:
            if pd.notna(skills) and skills:
                if isinstance(skills, str):
                    skills_list = [s.strip() for s in skills.split(",") if s.strip()]
                elif isinstance(skills, list):
                    skills_list = skills
                else:
                    continue
                cluster_counter.update(skills_list)

        # Calcul du score TF-IDF inversé
        scores = {}
        for skill, freq in cluster_counter.items():
            global_freq = global_counter.get(skill, 1)
            scores[skill] = freq / global_freq

        # Top compétences discriminantes
        if scores:
            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            labels[cid] = " / ".join([skill for skill, _ in top])
        else:
            labels[cid] = f"Cluster {cid}"

    return labels


# -----------------------------------
# CHARGEMENT DES DONNEES ET CALCULS
# -----------------------------------
with st.spinner("Chargement des données depuis MotherDuck..."):
    df = load_data(
        con,
        limit,
        contract_filter=contract_filter,
        date_filter=date_filter,
        region_filter=region_filter,
    )

st.caption(f"📄 {len(df)} offres analysées après application des filtres")

if len(df) == 0:
    st.warning(
        "⚠️ Aucune offre ne correspond aux critères sélectionnés. Veuillez ajuster vos filtres."
    )
    st.stop()

# Construction du texte ML
with st.spinner("Préparation des données textuelles..."):
    df["ml_text"] = df.apply(build_ml_text, axis=1)

# Chargement du modèle et calcul des embeddings
model = load_model()

with st.spinner("Calcul des embeddings sémantiques (768 dimensions)..."):
    embeddings = model.encode(
        df["ml_text"].tolist(),
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

# Réduction dimensionnelle UMAP
with st.spinner("Réduction dimensionnelle UMAP (768D → 3D)..."):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=3,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)

    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    df["z"] = coords[:, 2]

# Clustering HDBSCAN
with st.spinner("Clustering HDBSCAN en cours..."):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    df["cluster_id"] = clusterer.fit_predict(coords)

# Labellisation des clusters
cluster_labels_dict = label_clusters(df)
df["cluster_label"] = df["cluster_id"].map(cluster_labels_dict)

# ------------------------
# STATISTIQUES
# ------------------------
n_clusters = len(df[df["cluster_id"] != -1]["cluster_id"].unique())
n_noise = len(df[df["cluster_id"] == -1])

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🎯 Clusters identifiés", n_clusters)
with col2:
    st.metric("📊 Offres classées", len(df) - n_noise)
with col3:
    st.metric("🔍 Offres atypiques", n_noise)
with col4:
    pct_clustered = ((len(df) - n_noise) / len(df) * 100) if len(df) > 0 else 0
    st.metric("✅ Taux de clustering", f"{pct_clustered:.1f}%")

st.markdown("---")

# ------------------------
# VISUALISATION 3D
# ------------------------
st.markdown("### 🌐 Visualisation interactive 3D des clusters")

# Préparer les données pour l'affichage
df_display = df.copy()
df_display["hover_text"] = df_display.apply(
    lambda row: f"<b>{row['title']}</b><br>Cluster: {row['cluster_label']}<br>Compétences: {row['hard_skills'][:100] if pd.notna(row['hard_skills']) else 'N/A'}...",
    axis=1,
)

fig = px.scatter_3d(
    df_display,
    x="x",
    y="y",
    z="z",
    color="cluster_label",
    hover_data={
        "title": True,
        "cluster_label": True,
        "x": False,
        "y": False,
        "z": False,
    },
    opacity=0.7,
    color_discrete_sequence=px.colors.qualitative.Set3,
)

fig.update_traces(
    marker=dict(size=4, line=dict(width=0.5, color="white")),
    selector=dict(mode="markers"),
)

fig.update_layout(
    scene=dict(
        xaxis=dict(title="UMAP Dimension 1", visible=True),
        yaxis=dict(title="UMAP Dimension 2", visible=True),
        zaxis=dict(title="UMAP Dimension 3", visible=True),
        bgcolor="rgba(240,240,240,0.1)",
    ),
    showlegend=True,
    height=700,
    margin=dict(l=0, r=0, b=0, t=0),
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------
# DISTRIBUTION DES CLUSTERS
# ------------------------
st.markdown("### 📊 Distribution des offres par cluster")

cluster_counts = (
    df[df["cluster_id"] != -1]
    .groupby("cluster_label")
    .size()
    .sort_values(ascending=False)
)

fig_bar = px.bar(
    x=cluster_counts.values,
    y=cluster_counts.index,
    orientation="h",
    labels={"x": "Nombre d'offres", "y": "Cluster"},
    color=cluster_counts.values,
    color_continuous_scale="Viridis",
)

fig_bar.update_layout(
    showlegend=False,
    height=max(400, n_clusters * 40),
    yaxis={"categoryorder": "total ascending"},
)

st.plotly_chart(fig_bar, use_container_width=True)

# ------------------------
# DETAILS DES CLUSTERS
# ------------------------
with st.expander("📋 Détails des clusters"):
    for cid in sorted(df["cluster_id"].unique()):
        if cid == -1:
            continue

        cluster_data = df[df["cluster_id"] == cid]
        st.markdown(f"**Cluster {cid}: {cluster_labels_dict[cid]}**")
        st.caption(f"Taille: {len(cluster_data)} offres")

        sample_titles = cluster_data["title"].head(5).tolist()
        for i, title in enumerate(sample_titles, 1):
            st.text(f"  {i}. {title}")

        st.markdown("---")

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: #718096; font-size: 0.9rem;'>Powered by <strong>MotherDuck</strong> × <strong>Sentence Transformers</strong> × <strong>UMAP</strong> × <strong>HDBSCAN</strong> | RUCHE Team © 2026</div>",
    unsafe_allow_html=True,
)
