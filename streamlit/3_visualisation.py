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
        
# --- FONCTION DE CLUSTERING DES TITRES---
def cluster_job_titles(df, column='title'):
    # 1. Nettoyage et extraction des titres uniques pour optimiser le calcul
    unique_titles = df[column].unique().tolist()
    
    # 2. Embedding (Encodage sémantique)
    # On utilise un modèle multilingue performant pour le français
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(unique_titles, show_progress_bar=True)
    
    # 3. Réduction de dimension avec UMAP
    # On réduit à 5 dimensions pour aider HDBSCAN à trouver les densités
    reducer = umap.UMAP(
        n_neighbors=15, 
        n_components=5, 
        metric='cosine', 
        random_state=42
    )
    u_embeddings = reducer.fit_transform(embeddings)
    
    # 4. Clustering avec HDBSCAN
    # min_cluster_size : taille minimale pour former un groupe
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3, 
        metric='euclidean', 
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_labels(u_embeddings)
    
    # 5. Création d'une table de correspondance
    mapping_df = pd.DataFrame({
        column: unique_titles,
        'cluster_id': labels
    })
    
    # Nommer les clusters par leur titre le plus fréquent
    def get_representative_name(cluster_id):
        if cluster_id == -1: return "Autres / Non classé"
        # On prend le titre le plus court du cluster comme nom "propre"
        cluster_titles = mapping_df[mapping_df['cluster_id'] == cluster_id][column]
        return min(cluster_titles, key=len)

    cluster_names = {cid: get_representative_name(cid) for cid in set(labels)}
    mapping_df['titre_standardise'] = mapping_df['cluster_id'].map(cluster_names)
    
    # Fusionner avec le dataframe d'origine
    return df.merge(mapping_df[[column, 'titre_standardise']], on=column, how='left')
    
    # --- PRÉPARATION DES DONNÉES ---
@st.cache_data
df = load_data()
df['salaire_moyen'] = df['salaire'].apply(parse_salary_range)
# Suppression des lignes où le salaire n'a pas pu être extrait
df_clean = df.dropna(subset=['salaire_moyen'])
#Cluster des titres
df_clean = cluster_job_titles(df_clean)

# --- INTERFACE STREAMLIT ---
st.markdown("# Visualisation ")
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
    # Groupement par intitulé (on prend les 5 plus fréquents pour la lisibilité)
    top_jobs = df_clean['titre_standardise'].value_counts().nlargest(5).index
    df_top_jobs = df_clean[df_clean['titre_standardise'].isin(top_jobs)]
    
    avg_salary_cluster = df_salaires.groupby('titre_standardise')['salaire_moyen'].mean().sort_values(ascending=True).reset_index()

fig_salary = px.bar(
    avg_salary_cluster,
    x='salaire_moyen',
    y='titre_standardise',
    orientation='h',
    text_auto='.2s',
    labels={'salaire_moyen': 'Salaire Moyen (€)', 'titre_standardise': 'Métier (Cluster)'},
    color='salaire_moyen',
    color_continuous_scale='BuGn'
)
fig_salary.update_layout(yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig_salary, use_container_width=True)
    )
    st.plotly_chart(fig_salary, use_container_width=True)

with col_right:
    st.subheader(" Salaires par Région")
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
st.subheader(" Top 5 des compétences les plus demandées")
all_skills = df['hard_skills'].str.split(',').explode().str.strip()
top_skills = all_skills.value_counts().head(5).reset_index()
st.table(top_skills.rename(columns={'count': 'Nombre d\'offres', 'hard_skills': 'Compétence'}))

# 1. Préparation des données : on garde le lien entre le titre et les compétences
# Séparation des compétences en listes
df['skill_list'] = df['hard_skills'].str.split(',')

# "explode" sur la liste de compétences (le titre est dupliqué pour chaque compétence)
df_exploded = df.explode('skill_list')

# Nettoyage : espaces et mise en forme
df_exploded['skill_list'] = df_exploded['skill_list'].str.strip().str.capitalize()

# Suppression des lignes vides éventuelles
df_exploded = df_exploded[df_exploded['skill_list'] != ""]

# 2. Agrégation pour le graphique (on compte le nombre d'occurrences par métier et compétence)
top_7_skills = df_exploded['skill_list'].value_counts().nlargest(7).index
sunburst_data = (
    df_exploded[df_exploded['skill_list'].isin(top_7_skills)]
    .groupby(['title_standardisé', 'skill_list'])
    .size()
    .reset_index(name='Nombre d\'offres')
)

# 3. Création du graphique Sunburst
fig = px.sunburst(
    sunburst_data,
    path=['title_standardisé', 'skill_list'],
    values='Nombre d\'offres',
    color='title_standardisé',               
    title="Top compétences par intitulé de poste"
)

# Affichage dans Streamlit
st.plotly_chart(fig, use_container_width=True)

with st.container(border=False, horizontal=True, horizontal_alignment="center"):
    st.markdown(
        "<p style='text-align: center;'>Date de dernière Mise à jour de la BDD : DD//MM//YYYY</p>",
        unsafe_allow_html=True,
    )
