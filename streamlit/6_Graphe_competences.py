import networkx as nx
import streamlit as st
import itertools
import pandas as pd
import plotly.graph_objects as go
from collections import Counter
import os
from dotenv import load_dotenv
from ruche.db import get_connection
from pathlib import Path

# ------------------------
# CONNECTION DB
# ------------------------

con = get_connection()
st.set_page_config(
    layout="wide", page_title="Co-occurence de compétences", page_icon=":spider_web:"
)
CURRENT_DIR = Path(__file__).resolve().parent
LOGO_PATH = CURRENT_DIR / "static" / "Logo3.png"
with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 🐝 RUCHE")
        st.image(str(LOGO_PATH), width=140)
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------
# SIDEBAR : FILTRES
# --------------------

st.sidebar.markdown("## 🔍 Filtres")
limit = st.sidebar.slider(
    "Nombre d'offres analysées", min_value=200, max_value=5000, step=200, value=1000
)

min_weight = st.sidebar.slider("Seuil minimal de co-occurrence", 2, 20, 5)

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
# DATA
# ------------------------
@st.cache_data
def load_skills(
    _con, limit, contract_filter="Tous", date_filter="Toutes", region_filter="Toutes"
):
    """Charge les compétences depuis MotherDuck avec filtres appliqués"""

    query = f"""
        SELECT hard_skills
        FROM f_offre f
        LEFT JOIN d_localisation l ON f.id_ville = l.id_ville
        LEFT JOIN h_region r ON l.id_region = r.id_region
        LEFT JOIN d_contrat c ON f.id_contrat = c.id_contrat
        LEFT JOIN d_date d ON f.id_date_publication = d.id_date
        WHERE hard_skills IS NOT NULL
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


# -----------------------------------
# CHARGEMENT DES DONNÉES
# -----------------------------------
st.title("🕸️ Graphe de Co-occurrence des Compétences")
st.markdown(
    "Visualisation des synergies technologiques dans les offres d'emploi data/IA"
)
st.markdown("---")

df = load_skills(
    con,
    limit,
    contract_filter=contract_filter,
    date_filter=date_filter,
    region_filter=region_filter,
)

st.caption(f"📄 {len(df)} offres analysées après application des filtres")

# ------------------------
# BUILD GRAPH
# ------------------------
with st.spinner("Construction du graphe de co-occurrences..."):
    pairs = Counter()

    for skills in df["hard_skills"]:
        if skills and isinstance(skills, str):
            # Si c'est une chaîne, on la split
            skills_list = [s.strip() for s in skills.split(",") if s.strip()]
        elif skills and isinstance(skills, list):
            # Si c'est déjà une liste
            skills_list = skills
        else:
            continue

        if len(skills_list) > 1:
            for a, b in itertools.combinations(sorted(set(skills_list)), 2):
                pairs[(a, b)] += 1

    # Filtrer les paires par poids minimum
    edges = [(a, b, w) for (a, b), w in pairs.items() if w >= min_weight]

    # Créer le graphe
    G = nx.Graph()
    for a, b, w in edges:
        G.add_edge(a, b, weight=w)

# ------------------------
# LAYOUT ET VISUALISATION
# ------------------------
if len(G.nodes()) == 0:
    st.warning(
        "⚠️ Aucune co-occurrence trouvée avec ces paramètres. Essayez de diminuer le seuil minimal ou d'augmenter le nombre d'offres."
    )
else:
    with st.spinner("Calcul du layout 3D..."):
        pos = nx.spring_layout(G, dim=3, seed=42, k=0.5, iterations=50)

    # Extraire les coordonnées et créer les traces
    x_nodes, y_nodes, z_nodes, text_nodes, size_nodes = [], [], [], [], []

    for node in G.nodes():
        x_nodes.append(pos[node][0])
        y_nodes.append(pos[node][1])
        z_nodes.append(pos[node][2])
        text_nodes.append(f"{node}<br>Connexions: {G.degree(node)}")
        size_nodes.append(G.degree(node) * 3 + 5)

    # Créer les arêtes
    x_edges, y_edges, z_edges = [], [], []
    edge_weights = []

    for edge in G.edges(data=True):
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        x_edges.extend([x0, x1, None])
        y_edges.extend([y0, y1, None])
        z_edges.extend([z0, z1, None])
        edge_weights.append(edge[2]["weight"])

    # ------------------------
    # VISUALISATION
    # ------------------------
    fig = go.Figure()

    # Ajouter les arêtes
    fig.add_trace(
        go.Scatter3d(
            x=x_edges,
            y=y_edges,
            z=z_edges,
            mode="lines",
            line=dict(color="rgba(125,125,125,0.3)", width=1),
            hoverinfo="none",
            name="Relations",
        )
    )

    # Ajouter les nœuds
    fig.add_trace(
        go.Scatter3d(
            x=x_nodes,
            y=y_nodes,
            z=z_nodes,
            mode="markers+text",
            marker=dict(
                size=size_nodes,
                color=size_nodes,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Degré"),
                line=dict(color="white", width=0.5),
            ),
            text=text_nodes,
            hoverinfo="text",
            textposition="top center",
            name="Compétences",
        )
    )

    fig.update_layout(
        showlegend=False,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgba(240,240,240,0.1)",
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=700,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------
    # STATISTIQUES
    # ------------------------
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔵 Compétences", len(G.nodes()))
    with col2:
        st.metric("🔗 Relations", len(G.edges()))
    with col3:
        avg_degree = (
            sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0
        )
        st.metric("📊 Connexions moyennes", f"{avg_degree:.1f}")

    # ------------------------
    # TOP SYNERGIES
    # ------------------------
    st.markdown("### 🔝 Top 10 des synergies technologiques")
    top_edges = sorted(edges, key=lambda x: x[2], reverse=True)[:10]

    top_df = pd.DataFrame(
        top_edges, columns=["Compétence A", "Compétence B", "Co-occurrences"]
    )
    st.dataframe(top_df, use_container_width=True, hide_index=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: #718096; font-size: 0.9rem;'>Powered by <strong>MotherDuck</strong> × <strong>NetworkX</strong> | RUCHE Team © 2026</div>",
    unsafe_allow_html=True,
)
