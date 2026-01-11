import streamlit as st
import duckdb
import itertools
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from streamlit.config import MOTHERDUCK_DATABASE
from collections import Counter
import os
from dotenv import load_dotenv

# ------------------------
# CONNECTION DB
# ------------------------
dovenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dovenv_path)
MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
@st.cache_resource
def get_motherduck_connection():
    """Connexion au DataWhareHouse"""
    try:
        if not MOTHERDUCK_TOKEN:
            st.error(" Missing MotherDuck Token")
            st.stop()
        
        con = duckdb.connect(f"md:?motherduck_token={MOTHERDUCK_TOKEN}")
        con.execute(f"CREATE DATABASE IF NOT EXISTS {MOTHERDUCK_DATABASE}")
        con.close()
        con = duckdb.connect(f"md:{MOTHERDUCK_DATABASE}?motherduck_token={MOTHERDUCK_TOKEN}")
        
        return con
        
    except Exception as e:
        st.error(f"Erreur de connexion : {e}")
        st.stop()

con = get_motherduck_connection()
st.set_page_config(layout="wide", page_title="Co-occurence de compétences", page_icon=":spider_web:")
st.sidebar.image("./static/Logo3.png", width=150)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%); }
</style>
""", unsafe_allow_html=True)
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
# DATA
# ------------------------
@st.cache_data
def load_skills(con, limit, contract_filter='Tous', date_filter='Toutes', region_filter='Toutes'):
    
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

# -----------------------------------
# CREATION DU DATAFRAME
# -----------------------------------
df = load_skills(limit, 
                 contract_filteront=concract_filter, 
                 date_filter=date_filter, 
                 region_filter=region_filter 
                 )

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

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #718096; font-size: 0.9rem;'>Powered by <strong>MotherDuck</strong> × <strong>Sentence Transformers</strong> | RUCHE Team © 2026</div>", unsafe_allow_html=True)
