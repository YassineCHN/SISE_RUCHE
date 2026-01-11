import os
import streamlit as st
from sympy import re
import duckdb
import pandas as pd
import plotly.express as px
from streamlit.config import MOTHERDUCK_DATABASE
from dotenv import load_dotenv

# --- CHARGEMENT DES VARIABLES D'ENVIRONNEMENT ---
dovenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dovenv_path)
MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")


# --- CONNEXION MOTHERDUCK ---
@st.cache_resource
def get_motherduck_connection():
    """Connexion à MotherDuck"""
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
        st.error(f"❌ Erreur de connexion : {e}")
        st.stop()

db = get_motherduck_connection()
# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    query = """
    SELECT 
        job_id, 
        title, 
        salaire, 
        nom_region, 
        hard_skills, 
        id_date_publication
    FROM f_offre f
    LEFT JOIN d_localisation l ON f.id_ville = l.id_ville
    LEFT JOIN h_region r ON l.id_region = r.id_region
    LEFT JOIN d_contrat c ON f.id_contrat = c.id_contrat
    LEFT JOIN d_date d ON f.id_date_publication = d.id_date
    """
    return db.sql(query).df()



# --- TRAITEMENT DES COMPÉTENCES (Top 5) ---
def get_top_skills(dataframe):
    # On explose la colonne compétences si elle est stockée sous forme de liste ou texte
    all_skills = dataframe['hard_skills'].str.split(',').explode().str.strip()
    return all_skills.value_counts().head(5)

# --- FONCTION DE NETTOYAGE DES SALAIRES ---
def parse_salary_range(salary_str):
    if pd.isna(salary_str) or not isinstance(salary_str, str):
        return None
    
    # 1. Nettoyage de base : passage en minuscule et remplacement du 'k' et des espaces éventuels
    clean_str = salary_str.lower().replace(' ', '').replace('k', '000').replace('€', '')
    
    # Extraction de tous les nombres présents dans la chaîne
    numbers = re.findall(r'\d+', clean_str)
    if not numbers:
        return None
    
    # Convertir les extractions en float
    vals = [float(n) for n in numbers]

    # 2. Logique selon les opérateurs
    if '<' in clean_str:
        # Cas "< 25k€" -> On prend la borne haute
        return vals[0]
    
    elif '>' in clean_str:
        # Cas "> 100k€" -> On prend la borne basse
        return vals[0]
    
    elif '-' in clean_str or 'à' in clean_str:
        # Cas "25k - 30k" -> Moyenne de l'intervalle
        if len(vals) >= 2:
            return (vals[0] + vals[1]) / 2
        return vals[0]
    else:
        # Cas "40000" (valeur unique)
        return vals[0]
    
    # --- PRÉPARATION DES DONNÉES ---
df = load_data()
df['salaire_moyen'] = df['salaire'].apply(parse_salary_range)
# Suppression des lignes où le salaire n'a pas pu être extrait
df_clean = df.dropna(subset=['salaire_moyen'])

# --- INTERFACE STREAMLIT ---
st.markdown("# Visualisation 📊")
st.divider()
st.sidebar.markdown("# Visualisation du Marché de l'Emploi Data et IA")

# Row 1: KPIs
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Offres analysées", len(df_clean))
with col2:
    global_avg = df_clean['salaire_moyen'].mean()
    st.metric("Salaire Moyen Global", f"{global_avg:,.0f} €")
with col3:
    st.metric("Salaire Médian", f"{df_clean['salaire_moyen'].median():,.0f} €")

st.divider()
# Row 2: Graphiques
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(" Salaire moyen par intitulé de poste")
    # Groupement par intitulé (on prend les 10 plus fréquents pour la lisibilité)
    top_jobs = df_clean['title'].value_counts().nlargest(10).index
    df_top_jobs = df_clean[df_clean['title'].isin(top_jobs)]
    
    avg_salary_job = df_top_jobs.groupby('title')['salaire_moyen'].mean().sort_values(ascending=True).reset_index()
    
    fig_salary = px.bar(
        avg_salary_job,
        x='salaire_moyen',
        y='intitule',
        orientation='h',
        text_auto='.2s',
        labels={'salaire_moyen': 'Salaire Moyen (€)', 'intitule': 'Poste'},
        color='salaire_moyen',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_salary, use_container_width=True)

with col_right:
    st.subheader("📍 Salaires par Région")
    avg_salary_region = df_clean.groupby('region')['salaire_moyen'].mean().sort_values(ascending=False).reset_index()
    
    fig_region_sal = px.scatter(
        avg_salary_region,
        x='region',
        y='salaire_moyen',
        size='salaire_moyen',
        color='region',
        showlegend=False
    )
    st.plotly_chart(fig_region_sal, use_container_width=True)

# Row 3: Compétences (Rappel du besoin précédent)
st.subheader("🔝 Top 5 des compétences les plus demandées")
all_skills = df['hard_skills'].str.split(',').explode().str.strip()
top_skills = all_skills.value_counts().head(5).reset_index()
st.table(top_skills.rename(columns={'count': 'Nombre d\'offres', 'hard_skills': 'Compétence'}))


with st.container(border=False, horizontal=True, horizontal_alignment="center"):
    st.markdown(
        "<p style='text-align: center;'>Date de dernière Mise à jour de la BDD : DD//MM//YYYY</p>",
        unsafe_allow_html=True,
    )
