import streamlit as st
import duckdb
import itertools
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from streamlit.config import MOTHERDUCK_DATABASE
from collections import Counter

con = duckdb.connect("MOTHERDUCK_DATABASE")

# ------------------------
# titre
# ------------------------
st.title(" Graphe des co-occurrences de compétences")
# --------------------
# SIDEBAR : FILTRES
# --------------------

st.sidebar.markdown("##  Filtres")
limit = st.sidebar.slider(
    "Nombre d'offres analysées",
    min_value=200,
    max_value=5000,
    step=200,
    value=1000
)

min_weight = st.sidebar.slider(
    "Seuil minimal de co-occurrence",
    2,
    20,
    5
)
# Filtre contrat
st.sidebar.markdown("###  Type de contrat")
filter_cdi = st.sidebar.checkbox("CDI", value=False)
filter_cdd = st.sidebar.checkbox("CDD", value=False)
filter_stage = st.sidebar.checkbox("Stage", value=False)
filter_alternance = st.sidebar.checkbox("Alternance / Apprentissage", value=False)
filter_freelance = st.sidebar.checkbox("Freelance", value=False)
filter_interim = st.sidebar.checkbox("Intérim", value=False)



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
# DATA
# ------------------------
@st.cache_data
def load_skills(limit):
    con = duckdb.connect(MOTHERDUCK_DATABASE)
    query=f"""
        SELECT hard_skills
        FROM f_offre f
        LEFT JOIN d_localisation l ON f.id_ville = l.id_ville
        LEFT JOIN h_region r ON l.id_region = r.id_region
        LEFT JOIN d_contrat c ON f.id_contrat = c.id_contrat
        LEFT JOIN d_date d ON f.id_date_publication = d.id_date
        WHERE hard_skills IS NOT NULL
        LIMIT {limit}
        # Filtre contrat
    if contract_filters:
        conditions = []
        if contract_filters.get('cdi'):
            conditions.append("c.is_cdi = TRUE")
        if contract_filters.get('cdd'):
            conditions.append("c.is_cdd = TRUE")
        if contract_filters.get('stage'):
            conditions.append("c.is_stage = TRUE")
        if contract_filters.get('alternance'):
            conditions.append("c.is_apprentissage = TRUE")
        if contract_filters.get('freelance'):
            conditions.append("c.is_freelance = TRUE")
        if contract_filters.get('interim'):
            conditions.append("c.is_interim = TRUE")
        
        if conditions:
            query += "\n    AND (" + " OR ".join(conditions) + ")"
        
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

# -----------------------------------
# CHARGEMENT DES DONNEES ET CALCULS
# -----------------------------------
df = load_skills(contract_filters=contract_filters if any(contract_filters.values()) else None, date_filter, region_filter, limit)

# ------------------------
# BUILD GRAPH
# ------------------------
pairs = Counter()

for skills in df["hard_skills"]:
    if skills and len(skills) > 1:
        for a, b in itertools.combinations(sorted(set(skills)), 2):
            pairs[(a, b)] += 1

edges = [
    (a, b, w)
    for (a, b), w in pairs.items()
    if w >= min_weight
]

G = nx.Graph()
for _, row in edges.iterrows():
    G.add_edge(row.skill_1, row.skill_2, weight=row.weight)

# ------------------------
# LAYOUT
# ------------------------
pos = nx.spring_layout(G, dim=3, seed=42)
x, y, z, text = [], [], [], []
for node in G.nodes():
    x.append(pos[node][0])
    y.append(pos[node][1])
    z.append(pos[node][2])
    text.append(node)
    size.append(G.degree(node) * 2)

# ------------------------
# VISUALISATION
# ------------------------
fig = go.Figure(
    data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(size=size),
            text=text,
            hoverinfo="text"
        )
    ]
)

st.plotly_chart(fig, use_container_width=True)
st.caption(f"{len(G.nodes())} compétences – {len(G.edges())} relations")
