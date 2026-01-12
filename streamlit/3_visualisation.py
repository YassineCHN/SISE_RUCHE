import os
import streamlit as st
import re
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from pathlib import Path

from ruche.db import get_connection


st.set_page_config(
    layout="wide", page_title="Visualisation des données", page_icon="💼"
)

CURRENT_DIR = Path(__file__).resolve().parent
LOGO_PATH = CURRENT_DIR / "static" / "Logo3.png"

with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 🐝 RUCHE")
        st.image(str(LOGO_PATH), width=140)
    st.markdown("## 📊 Visualisation")
    st.caption(
        "Analyse exploratoire du marché de l’emploi Data & IA "
        "à partir des offres collectées et enrichies."
    )

    st.divider()

    st.markdown("### 🔍 Ce que vous voyez")
    st.markdown(
        "- Distribution des **salaires** par métier\n"
        "- Comparaison **régionale**\n"
        "- Analyse des **compétences techniques**\n"
        "- Clustering sémantique des intitulés"
    )

    st.divider()

    st.markdown("### 💡 Comment lire les graphiques")
    st.markdown(
        "- Les barres représentent des **moyennes**\n"
        "- Les tailles indiquent le **volume d’offres**\n"
        "- Les clusters regroupent des métiers similaires"
    )
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
</style>
""",
    unsafe_allow_html=True,
)

# --- CONNEXION --
db = get_connection()


# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data(_db):
    """Charge les données depuis MotherDuck"""
    query = """
    SELECT 
        f.job_id, 
        f.title, 
        f.salaire, 
        r.nom_region, 
        f.hard_skills, 
        f.id_date_publication
    FROM f_offre f
    LEFT JOIN d_localisation l ON f.id_ville = l.id_ville
    LEFT JOIN h_region r ON l.id_region = r.id_region
    LEFT JOIN d_contrat c ON f.id_contrat = c.id_contrat
    LEFT JOIN d_date d ON f.id_date_publication = d.id_date
    WHERE f.salaire IS NOT NULL
    """
    return _db.execute(query).df()


# --- FONCTION DE NETTOYAGE DES SALAIRES ---
def parse_salary_range(salary_str):
    """Parse une chaîne de salaire et retourne un float représentant la moyenne"""
    if pd.isna(salary_str) or not isinstance(salary_str, str):
        return None

    # Nettoyage de base
    clean_str = (
        salary_str.lower()
        .replace(" ", "")
        .replace("k", "000")
        .replace("€", "")
        .replace(",", "")
    )

    # Extraction de tous les nombres
    numbers = re.findall(r"\d+", clean_str)
    if not numbers:
        return None

    # Convertir en float
    vals = [float(n) for n in numbers]

    # Logique selon les opérateurs
    if "<" in clean_str:
        return vals[0]
    elif ">" in clean_str:
        return vals[0]
    elif "-" in clean_str or "à" in clean_str:
        if len(vals) >= 2:
            return (vals[0] + vals[1]) / 2
        return vals[0]
    else:
        return vals[0]


# --- FONCTION DE CLUSTERING DES TITRES ---
@st.cache_data
def cluster_job_titles(df, column="title"):
    """Cluster les titres de poste similaires en groupes sémantiques"""

    # Extraction des titres uniques
    unique_titles = df[column].dropna().unique().tolist()

    if len(unique_titles) == 0:
        df["titre_standardise"] = df[column]
        return df

    if len(unique_titles) < 5:
        # Pas assez de données pour faire un clustering
        df["titre_standardise"] = df[column]
        return df

    # Embedding
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(unique_titles, show_progress_bar=False)

    # Réduction de dimension avec UMAP
    n_neighbors = min(15, len(unique_titles) - 1)
    n_components = min(5, len(unique_titles) - 1)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        random_state=42,
    )
    u_embeddings = reducer.fit_transform(embeddings)

    # Clustering avec HDBSCAN
    min_cluster_size = max(3, len(unique_titles) // 20)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(u_embeddings)

    # Création d'une table de correspondance
    mapping_df = pd.DataFrame({column: unique_titles, "cluster_id": labels})

    # Nommer les clusters
    def get_representative_name(cluster_id):
        if cluster_id == -1:
            return "Autres / Non classé"
        cluster_titles = mapping_df[mapping_df["cluster_id"] == cluster_id][column]
        if len(cluster_titles) == 0:
            return f"Cluster {cluster_id}"
        return min(cluster_titles, key=len)

    cluster_names = {cid: get_representative_name(cid) for cid in set(labels)}
    mapping_df["titre_standardise"] = mapping_df["cluster_id"].map(cluster_names)

    # Fusionner avec le dataframe d'origine
    return df.merge(mapping_df[[column, "titre_standardise"]], on=column, how="left")


# --- INTERFACE STREAMLIT ---
st.title("📊 Visualisation du Marché de l'Emploi Data et IA")
st.markdown("Analyse des salaires, compétences et répartition géographique des offres")
st.markdown("---")

# --- PRÉPARATION DES DONNÉES ---
with st.spinner("Chargement des données depuis MotherDuck..."):
    df = load_data(db)

with st.spinner("Traitement des salaires..."):
    df["salaire_moyen"] = df["salaire"].apply(parse_salary_range)
    df_clean = df.dropna(subset=["salaire_moyen"])

with st.spinner("Clustering des titres de poste..."):
    df_clean = cluster_job_titles(df_clean)

st.caption(f"📄 {len(df_clean)} offres analysées avec salaires valides")

# --- ROW 1: KPIs ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📋 Offres", f"{len(df_clean):,}")

with col2:
    global_avg = df_clean["salaire_moyen"].mean()
    st.metric("💰 Salaire Moyen", f"{global_avg:,.0f} €")

with col3:
    median_salary = df_clean["salaire_moyen"].median()
    st.metric("📊 Salaire Médian", f"{median_salary:,.0f} €")

with col4:
    n_regions = df_clean["nom_region"].nunique()
    st.metric("🗺️ Régions", n_regions)

st.markdown("---")

# --- ROW 2: GRAPHIQUES SALAIRES ---
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("💼 Salaire moyen par intitulé de poste")

    # Groupement par titre standardisé
    avg_salary_cluster = (
        df_clean.groupby("titre_standardise")["salaire_moyen"]
        .agg(["mean", "count"])
        .reset_index()
    )
    avg_salary_cluster.columns = ["titre_standardise", "salaire_moyen", "count"]

    # Filtrer les clusters avec au moins 3 offres
    avg_salary_cluster = avg_salary_cluster[avg_salary_cluster["count"] >= 3]

    # Garder les 10 plus hauts salaires
    avg_salary_cluster = avg_salary_cluster.nlargest(10, "salaire_moyen")

    if len(avg_salary_cluster) > 0:
        fig_salary = px.bar(
            avg_salary_cluster,
            x="salaire_moyen",
            y="titre_standardise",
            orientation="h",
            text_auto=".0f",
            labels={
                "salaire_moyen": "Salaire Moyen (€)",
                "titre_standardise": "Métier",
            },
            color="salaire_moyen",
            color_continuous_scale="BuGn",
        )
        fig_salary.update_layout(
            yaxis={"categoryorder": "total ascending"}, showlegend=False, height=400
        )
        fig_salary.update_traces(texttemplate="%{x:,.0f} €", textposition="outside")
        st.plotly_chart(fig_salary, width="stretch")
    else:
        st.info("Pas assez de données pour afficher les salaires par métier")

with col_right:
    st.subheader("🗺️ Salaires par Région")

    avg_salary_region = (
        df_clean.groupby("nom_region")["salaire_moyen"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    if len(avg_salary_region) > 0:
        fig_region_sal = px.scatter(
            avg_salary_region,
            x="nom_region",
            y="salaire_moyen",
            size="salaire_moyen",
            color="nom_region",
            labels={"nom_region": "Région", "salaire_moyen": "Salaire Moyen (€)"},
        )
        fig_region_sal.update_layout(
            showlegend=False, xaxis={"tickangle": 45}, height=400
        )
        st.plotly_chart(fig_region_sal, width="stretch")
    else:
        st.info("Pas de données régionales disponibles")

st.markdown("---")

# --- ROW 3: COMPÉTENCES ---
st.subheader("🔧 Analyse des compétences techniques")

# Traitement des compétences
df_clean_skills = df_clean[df_clean["hard_skills"].notna()].copy()

if len(df_clean_skills) > 0:
    all_skills = (
        df_clean_skills["hard_skills"]
        .str.split(",")
        .explode()
        .str.strip()
        .str.capitalize()
    )
    all_skills = all_skills[all_skills != ""]

    top_skills = all_skills.value_counts().head(5).reset_index()
    top_skills.columns = ["Compétence", "Nombre d'offres"]

    col_table, col_sunburst = st.columns([1, 2])

    with col_table:
        st.markdown("**Top 5 des compétences**")
        st.dataframe(top_skills, width="stretch", hide_index=True)

    with col_sunburst:
        # Préparation des données pour le sunburst
        df_exploded = df_clean_skills.copy()
        df_exploded["skill_list"] = df_exploded["hard_skills"].str.split(",")
        df_exploded = df_exploded.explode("skill_list")
        df_exploded["skill_list"] = (
            df_exploded["skill_list"].str.strip().str.capitalize()
        )
        df_exploded = df_exploded[df_exploded["skill_list"].notna()]
        df_exploded = df_exploded[df_exploded["skill_list"] != ""]

        # Top 7 compétences
        top_7_skills = df_exploded["skill_list"].value_counts().nlargest(7).index
        sunburst_data = (
            df_exploded[df_exploded["skill_list"].isin(top_7_skills)]
            .groupby(["titre_standardise", "skill_list"])
            .size()
            .reset_index(name="Nombre d'offres")
        )

        # Graphique Sunburst
        if len(sunburst_data) > 0:
            fig_sunburst = px.sunburst(
                sunburst_data,
                path=["titre_standardise", "skill_list"],
                values="Nombre d'offres",
                color="titre_standardise",
                title="Compétences par métier (Top 7)",
            )
            fig_sunburst.update_layout(height=500)
            st.plotly_chart(fig_sunburst, width="stretch")
        else:
            st.info("Pas assez de données pour générer le graphique sunburst")
else:
    st.info("Aucune donnée de compétences disponible")

st.markdown("---")

# --- FOOTER ---
st.markdown(
    "<p style='text-align: center; color: #718096; margin-top: 2rem;'>Date de dernière mise à jour : Données en temps réel</p>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='text-align: center; color: #718096; font-size: 0.9rem; margin-top: 1rem;'>"
    "Powered by <strong>MotherDuck</strong> × <strong>Sentence Transformers</strong> × <strong>UMAP</strong> × <strong>HDBSCAN</strong> | RUCHE Team © 2026"
    "</div>",
    unsafe_allow_html=True,
)
