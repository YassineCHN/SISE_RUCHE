"""
Cartographie des Offres d'Emploi
"""
import streamlit as st
import pandas as pd
import duckdb
import folium
from streamlit_folium import st_folium
import plotly.express as px 
import math
from collections import Counter
import sys
import os
from dotenv import load_dotenv

from config import MOTHERDUCK_DATABASE
dovenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dovenv_path)
MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")

st.set_page_config(page_title="Cartographie", page_icon="🗺️", layout="wide")

# ============================================================================
# CONNEXION MOTHERDUCK
# ============================================================================

@st.cache_resource
def get_motherduck_connection():
    """Connexion à MotherDuck"""
    try:
        if not MOTHERDUCK_TOKEN:
            st.error("❌ Token MotherDuck manquant")
            st.stop()
        
        con = duckdb.connect(f"md:?motherduck_token={MOTHERDUCK_TOKEN}")
        con.execute(f"CREATE DATABASE IF NOT EXISTS {MOTHERDUCK_DATABASE}")
        con.close()
        con = duckdb.connect(f"md:{MOTHERDUCK_DATABASE}?motherduck_token={MOTHERDUCK_TOKEN}")
        
        return con
        
    except Exception as e:
        st.error(f"❌ Erreur de connexion : {e}")
        st.stop()

conn = get_motherduck_connection()

# ============================================================================
# RÉCUPÉRATION DES VALEURS UNIQUES POUR LES FILTRES
# ============================================================================

@st.cache_data(ttl=3600)
def get_unique_values(_conn):
    """Récupère les valeurs uniques pour les filtres multisélection"""
    
    # Hard Skills (extraire depuis le champ texte)
    hard_skills_query = """
    SELECT DISTINCT 
        UNNEST(string_split(f.hard_skills, ',')) as skill
    FROM f_offre f
    WHERE f.is_duplicate = FALSE 
        AND f.hard_skills IS NOT NULL 
        AND f.hard_skills != ''
    ORDER BY skill
    """
    
    hard_skills_df = _conn.execute(hard_skills_query).fetchdf()
    hard_skills_list = [str(skill).strip() for skill in hard_skills_df['skill'].tolist() if skill]
    hard_skills_list = sorted(list(set(hard_skills_list)))  # Dédupliquer et trier
    
    # Job Functions (extraire depuis le champ texte)
    job_function_query = """
    SELECT DISTINCT 
        UNNEST(string_split(f.job_function, ',')) as fonction
    FROM f_offre f
    WHERE f.is_duplicate = FALSE 
        AND f.job_function IS NOT NULL 
        AND f.job_function != ''
    ORDER BY fonction
    """
    
    job_function_df = _conn.execute(job_function_query).fetchdf()
    job_function_list = [str(func).strip() for func in job_function_df['fonction'].tolist() if func]
    job_function_list = sorted(list(set(job_function_list)))  # Dédupliquer et trier
    
    return hard_skills_list, job_function_list

# Charger les valeurs uniques
hard_skills_available, job_functions_available = get_unique_values(conn)

# ============================================================================
# SIDEBAR : FILTRES
# ============================================================================

st.sidebar.markdown("## 🔍 Filtres")

# Filtre contrat
st.sidebar.markdown("### 📋 Type de contrat")
filter_cdi = st.sidebar.checkbox("CDI", value=False)
filter_cdd = st.sidebar.checkbox("CDD", value=False)
filter_stage = st.sidebar.checkbox("Stage", value=False)
filter_alternance = st.sidebar.checkbox("Alternance / Apprentissage", value=False)
filter_freelance = st.sidebar.checkbox("Freelance", value=False)
filter_interim = st.sidebar.checkbox("Intérim", value=False)
filter_public = st.sidebar.checkbox("Contrat public", value=False)


# Filtre salaire
st.sidebar.markdown("### 💰 Salaire")
salary_filter = st.sidebar.radio(
    "Fourchette",
    options=[
        'Tous',
        'Renseigné',
        '< 25k€',
        '25k€ - 30k€',
        '30k€ - 35k€',
        '35k€ - 40k€',
        '40k€ - 45k€',
        '45k€ - 50k€',
        '50k€ - 60k€',
        '60k€ - 70k€',
        '70k€ - 80k€',
        '80k€ - 100k€',
        '> 100k€',
        'A négocier'
    ],
    index=0
)

# Filtre date
st.sidebar.markdown("### 📅 Date de publication")
date_filter = st.sidebar.radio(
    "Publié depuis",
    options=['Toutes', '7 jours', '21 jours', '1 mois', '3 mois'],
    index=0
)

# Filtre Hard Skills
st.sidebar.markdown("### 🛠️ Compétences techniques")
selected_hard_skills = st.sidebar.multiselect(
    "Hard Skills",
    options=hard_skills_available,
    default=[],
    placeholder="Sélectionnez des compétences...",
    help="Sélectionnez une ou plusieurs compétences techniques"
)

# Filtre Job Function
st.sidebar.markdown("### 💼 Fonction")
selected_job_functions = st.sidebar.multiselect(
    "Job Function",
    options=job_functions_available,
    default=[],
    placeholder="Sélectionnez des fonctions...",
    help="Sélectionnez une ou plusieurs fonctions métier"
)

# Bouton reset
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Réinitialiser les filtres", use_container_width=True):
    st.rerun()

# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

@st.cache_data(ttl=600)
def load_map_data(_conn, contract_filters=None, salary_filter='Tous', date_filter='Toutes', 
                  hard_skills=None, job_functions=None):
    """Charge les données avec filtres appliqués"""

    query = """
    SELECT 
        l.ville,
        l.latitude,
        l.longitude,
        l.departement,
        r.nom_region,
        
        COUNT(*) as nb_offres,
        
        ARRAY_AGG(f.job_id) as job_ids,
        ARRAY_AGG(f.title) as titles,
        ARRAY_AGG(f.company_name) as companies,
        ARRAY_AGG(c.type_contrat) as contracts,
        ARRAY_AGG(f.salaire) as salaries,
        ARRAY_AGG(f.source_url) as urls
        
    FROM f_offre f
    LEFT JOIN d_localisation l ON f.id_ville = l.id_ville
    LEFT JOIN h_region r ON l.id_region = r.id_region
    LEFT JOIN d_contrat c ON f.id_contrat = c.id_contrat
    LEFT JOIN d_date d ON f.id_date_publication = d.id_date
    
    WHERE 
        f.is_duplicate = FALSE
        AND l.latitude IS NOT NULL
        AND l.longitude IS NOT NULL
    """
    
    # Filtre contrat
    # Filtre contrat (nouveau modèle: d_contrat.type_contrat)
    if contract_filters:
        selected_contracts = [k for k, v in contract_filters.items() if v]

        if selected_contracts:
            # protection simple contre quotes (même si ici ce sont des constantes)
            selected_contracts = [c.replace("'", "''") for c in selected_contracts]
            contracts_sql = ", ".join(f"'{c}'" for c in selected_contracts)
            query += f"\n    AND c.type_contrat IN ({contracts_sql})"

    # Filtre salaire - utilisation de catégorie_salaire
    if salary_filter == 'Renseigné':
        query += "\n    AND f.salaire IS NOT NULL AND f.salaire != '' AND f.salaire != 'Non spécifié'"
    elif salary_filter != 'Tous':
        # Correspondance directe avec la catégorie
        query += f"\n    AND f.salaire = '{salary_filter}'"
    
    # Filtre date
    if date_filter == '7 jours':
        query += "\n    AND d.date_complete >= CURRENT_DATE - INTERVAL '7 days'"
    elif date_filter == '21 jours':
        query += "\n    AND d.date_complete >= CURRENT_DATE - INTERVAL '21 days'"
    elif date_filter == '1 mois':
        query += "\n    AND d.date_complete >= CURRENT_DATE - INTERVAL '30 days'"
    elif date_filter == '3 mois':
        query += "\n    AND d.date_complete >= CURRENT_DATE - INTERVAL '90 days'"
    
    # Filtre Hard Skills
    if hard_skills and len(hard_skills) > 0:
        skills_conditions = []
        for skill in hard_skills:
            safe_skill = skill.replace("'", "''")
            skills_conditions.append(f"f.hard_skills ILIKE '%{safe_skill}%'")

        query += "\n    AND (" + " OR ".join(skills_conditions) + ")"
    
    # Filtre Job Function
    if job_functions and len(job_functions) > 0:
        functions_conditions = []
        for func in job_functions:
            functions_conditions.append(f"f.job_function LIKE '%{func}%'")
        
        if functions_conditions:
            query += "\n    AND (" + " OR ".join(functions_conditions) + ")"

    query += """
    GROUP BY l.ville, l.latitude, l.longitude, l.departement, r.nom_region
    ORDER BY nb_offres DESC
    LIMIT 500
    """
    
    df = _conn.execute(query).fetchdf()
    return df

# Préparer les filtres
contract_filters = {
    "CDI": filter_cdi,
    "CDD": filter_cdd,
    "STAGE": filter_stage,
    "ALTERNANCE": filter_alternance,
    "INTERIM": filter_interim,
    "AUTRE": filter_freelance,  # ← ton checkbox "Freelance" mappe vers AUTRE
    "CONTRAT_PUBLIC": filter_public,
}
# Charger les données avec filtres
df = load_map_data(
    conn, 
    contract_filters=contract_filters if any(contract_filters.values()) else None,
    salary_filter=salary_filter,
    date_filter=date_filter,
    hard_skills=selected_hard_skills if selected_hard_skills else None,
    job_functions=selected_job_functions if selected_job_functions else None
)

if df.empty:
    st.warning("⚠️ Aucune offre ne correspond aux filtres sélectionnés")
    st.stop()

# ============================================================================
# EN-TÊTE
# ============================================================================

st.markdown("# 🗺️ Cartographie des Offres")

# ============================================================================
# MÉTRIQUES + STATISTIQUES COMPACTES
# ============================================================================

total_offres = int(df['nb_offres'].sum())
nb_villes = len(df)
nb_regions = df['nom_region'].nunique()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📌 Offres", f"{total_offres:,}")
with col2:
    st.metric("🏙️ Villes", nb_villes)
with col3:
    st.metric("🗺️ Régions", nb_regions)

# ────────────────────────────────────────────────────────────────────────
# TOP 4 RÉGIONS
# ────────────────────────────────────────────────────────────────────────

st.markdown("---")

region_stats = df.groupby('nom_region')['nb_offres'].sum().sort_values(ascending=False).head(4)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌍 Top 4 Régions")
    for idx, (region, nb) in enumerate(region_stats.items(), 1):
        pct = (nb / total_offres) * 100
        st.markdown(f"**{idx}. {region}**  \n{int(nb):,} offres ({pct:.1f}%)")

# ────────────────────────────────────────────────────────────────────────
# TOP 4 VILLES
# ────────────────────────────────────────────────────────────────────────

with col2:
    st.markdown("### 🏙️ Top 4 Villes")
    top_cities = df.nlargest(4, 'nb_offres')
    
    for idx, row in enumerate(top_cities.iterrows(), 1):
        _, city = row
        pct = (city['nb_offres'] / total_offres) * 100
        st.markdown(f"**{idx}. {city['ville']}** ({city['departement']})  \n{int(city['nb_offres']):,} offres ({pct:.1f}%)")

st.markdown("---")

# ============================================================================
# CARTE INTERACTIVE AVEC CLUSTERING DYNAMIQUE
# ============================================================================

st.markdown("### 🗺️ Carte interactive")
st.caption("💡 Les points se regroupent automatiquement selon le zoom. Cliquez sur un cluster pour zoomer.")

# Centre de la carte
center_lat = df['latitude'].mean()
center_lon = df['longitude'].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=6,
    tiles='OpenStreetMap'
)

# ────────────────────────────────────────────────────────────────────────
# AJOUTER LE PLUGIN MARKERCLUSTER
# ────────────────────────────────────────────────────────────────────────

from folium.plugins import MarkerCluster

# Créer le cluster avec options personnalisées
marker_cluster = MarkerCluster(
    name='Offres',
    overlay=True,
    control=True,
    icon_create_function="""
    function(cluster) {
        var childCount = cluster.getChildCount();
        var c = ' marker-cluster-';
        
        // Gradient de couleur basé sur le nombre d'offres
        if (childCount < 10) {
            c += 'small';
        } else if (childCount < 50) {
            c += 'medium';
        } else {
            c += 'large';
        }
        
        return new L.DivIcon({ 
            html: '<div><span>' + childCount + '</span></div>', 
            className: 'marker-cluster' + c, 
            iconSize: new L.Point(40, 40) 
        });
    }
    """
).add_to(m)

# Fonction gradient heatmap (conservée pour les marqueurs individuels)
def get_heatmap_color(nb_offres, max_offres):
    """Gradient en fonction du nombre d'offres"""
    normalized = nb_offres / max_offres if max_offres > 0 else 0
    
    if normalized < 0.02:
        return "#d8a243"   # Marron foncé
    elif normalized < 0.10:
        return "#a91b1b"  # Rouge foncé
    else:
        return "#4e1111"  # Jaune

max_offres = df['nb_offres'].max()

# ────────────────────────────────────────────────────────────────────────
# AJOUTER LES MARQUEURS AU CLUSTER
# ────────────────────────────────────────────────────────────────────────

for _, row in df.iterrows():
    ville = row['ville']
    departement = row['departement']
    region = row['nom_region']
    nb_offres = int(row['nb_offres'])
    
    job_ids = row['job_ids']
    titles = row['titles']
    companies = row['companies']
    contracts = row['contracts']
    salaries = row['salaries']
    urls = row['urls']
    
    # ────────────────────────────────────────────────────────────────────
    # TOOLTIP (survol)
    # ────────────────────────────────────────────────────────────────────
    tooltip = f"""
    <div style='font-family: Arial; font-size: 13px;'>
        <b style='font-size: 15px;'>{ville}</b><br>
        📍 {departement} - {region}<br>
        📌 <b>{nb_offres}</b> offre{'s' if nb_offres > 1 else ''}
    </div>
    """
    
    # ────────────────────────────────────────────────────────────────────
    # POPUP (clic)
    # ────────────────────────────────────────────────────────────────────
    popup_html = f"""
    <div style="width: 400px; max-height: 500px; overflow-y: auto; font-family: Arial;">
        <div style="position: sticky; top: 0; background: white; padding: 10px 0; 
                    border-bottom: 3px solid #1f77b4; margin-bottom: 10px;">
            <h3 style="margin: 0; color: #1f77b4;">📍 {ville} ({departement})</h3>
            <p style="margin: 5px 0; color: #666;">{region} • {nb_offres} offres</p>
        </div>
    """
    
    # Limiter à 30 offres dans le popup
    for i in range(min(30, nb_offres)):
        title = titles[i][:50] if titles[i] else 'Sans titre'
        company = companies[i][:30] if companies[i] else 'N/A'
        contract = contracts[i] if contracts[i] else 'N/A'
        salary = salaries[i] if salaries[i] else 'Non spécifié'
        url = urls[i] if urls[i] else '#'
        
        popup_html += f"""
        <div style="border-left: 4px solid #1f77b4; padding: 8px; margin: 8px 0; 
                    background: #f8f9fa; border-radius: 4px;">
            <p style="margin: 0 0 4px 0; font-weight: bold; font-size: 13px;">
                {title}{'...' if len(titles[i] or '') > 50 else ''}
            </p>
            <p style="margin: 2px 0; font-size: 11px; color: #555;">
                🏢 {company}{'...' if len(companies[i] or '') > 30 else ''}
            </p>
            <p style="margin: 2px 0; font-size: 11px; color: #555;">
                📋 {contract} | 💰 {salary}
            </p>
            <a href="{url}" target="_blank" 
               style="display: inline-block; margin-top: 4px; padding: 4px 10px; 
                      background: #28a745; color: white; text-decoration: none; 
                      border-radius: 3px; font-size: 11px;">
                🔗 Voir l'offre
            </a>
        </div>
        """
    
    if nb_offres > 30:
        popup_html += f"""
        <p style="text-align: center; color: #999; font-style: italic; padding: 10px;">
            ... et {nb_offres - 30} autre(s) offre(s)
        </p>
        """
    
    popup_html += "</div>"
    
    # ────────────────────────────────────────────────────────────────────
    # STYLE DU MARQUEUR (HEATMAP)
    # ────────────────────────────────────────────────────────────────────
    
    # Couleur selon intensité (heatmap)
    marker_color = get_heatmap_color(nb_offres, max_offres)
    
    # Taille proportionnelle (logarithmique)
    radius = min(10 + math.log(nb_offres + 1) * 2, 30)
    
    # Créer le marqueur et l'ajouter AU CLUSTER (pas à la carte directement)
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=radius,
        popup=folium.Popup(popup_html, max_width=450),
        tooltip=folium.Tooltip(tooltip),
        color=marker_color,
        fill=True,
        fillColor=marker_color,
        fillOpacity=0.6,
        weight=2
    ).add_to(marker_cluster)  # ← AJOUTÉ AU CLUSTER

# Afficher la carte
st_folium(m, width=None, height=650, use_container_width=True)

# ============================================================================
# MESSAGE FINAL
# ============================================================================

st.markdown("---")
st.success(f"✅ **Carte prête** : {total_offres:,} offres affichées sur {nb_villes} villes")