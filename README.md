# 🐝 RUCHE
Projet NLP & Text Mining – Master 2 SISE (2025–2026) 

### Application de cherche d'emplois

<p align="center">
  <img src="streamlit/static/Logo3.png" alt="RUCHE" width="280">
</p>

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-DuckDB-2E86C1?style=for-the-badge&logo=databricks&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Data%20Lake-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![MotherDuck](https://img.shields.io/badge/MotherDuck-DuckDB%20Cloud-FFD43B?style=for-the-badge&logo=duckdb&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Folium](https://img.shields.io/badge/Folium-Maps-77B829?style=for-the-badge)
![Selenium](https://img.shields.io/badge/Selenium-Web%20Scraping-43B02A?style=for-the-badge&logo=selenium&logoColor=white)
![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup-HTML%20Parsing-59666C?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Mistral](https://img.shields.io/badge/Mistral-LLM%20API-FFB703?style=for-the-badge)


## Présentation du projet

**RUCHE** est une plateforme d’analyse du marché de l’emploi **Data Science & Intelligence Artificielle** en France.  
Elle combine **web scraping**, **NLP**, **machine learning**, **data warehousing** et **visualisation interactive** pour proposer :

-  **Une recherche sémantique intelligente** d’offres d’emploi  
-  **Une cartographie géographique interactive** du marché de l’emploi  
-  **Des analyses avancées** sur les salaires, les compétences et les tendances du marché
-  **Enregistrement de nouvelles offres** pour les utilisateurs de l'application

Le système repose sur une **architecture end-to-end**, depuis la collecte des données jusqu’à leur exploitation analytique au sein d’une application **Streamlit**.

---

## 🧠 Objectifs du projet

Le projet RUCHE s’inscrit dans le cadre du module **NLP & Text Mining** du Master 2 SISE et répond aux objectifs pédagogiques suivants :

- 📥 **Constituer un corpus d’offres d’emploi**
  - Extraction automatisée d’annonces issues de plateformes d’emploi accessibles en ligne  
    (France Travail, APEC, JobTeaser, Choisir le Service Public, etc.)
  - Collecte réalisée via des techniques de **web scraping** (BeautifulSoup, Selenium) et des **API** lorsque disponibles
  - Exploitation des champs structurés lorsqu’ils sont disponibles  
    *(titre, missions, compétences, profil, rémunération, localisation, type de contrat…)*
  - Analyse du **corps textuel complet** lorsque la structure est absente ou hétérogène
  - Focalisation sur les **métiers et compétences liés à la Data Science et à l’Intelligence Artificielle**
  - Stocker sur MongoDB (Base NoSql) dans différentes collections les offres scrapper
  - 6000 offres collectés 

- 🗄️ **Mettre en place un entrepôt de données**
  - Créaction d'une pipeline d'ETL pour **extraire** nos offre de MongoDb, les **transformer** et les **charger** dans une BDD relationnel sur MotherDuckdb
  - Modélisation sous forme de **schéma en étoile** (table de faits et dimensions)
  - Stockage dans un **SGBD libre** (DuckDB via MotherDuck)
  - Connexion directe entre l’application et la base de données analytique
  - ~4000 offres après nettoyages stocker sur MotherDuck et DuckDB
    
- 🧠 **Appliquer des méthodes avancées de NLP et de Machine Learning**
  - Filtrage automatique des offres non pertinentes (hors data / IA)
  - Vectorisation sémantique des annonces
  - Recherche par similarité en langage naturel
  - Analyses interprétables et lisibles, y compris lors de l’usage de modèles de langage (LLM)

- 🌐 **Développer une application web interactive**
  - Application Python basée sur **Streamlit**
  - Interface dédiée à l’exploration, la recherche et l’analyse du corpus
  - Visualisations interactives (cartes, graphiques dynamiques, clustering)

- 🗺️ **Intégrer une dimension géographique**
  - Analyse territoriale à l’échelle des villes, départements et régions
  - Représentations cartographiques interactives

- ➕ **Permettre l’ajout dynamique de nouvelles offres**
  - Ajout manuel ou semi-automatisé d’annonces (LLM - Mistral)
  - Mécanismes de **détection de doublons** pour préserver la qualité du corpus

- 🚢 **Garantir la reproductibilité et le déploiement**
  - Déploiement de l’ensemble du système via une **image Docker**
  - L’utilisateur peut lancer l’application sans configuration complexe

---

## 🏗️ Architecture globale

```
┌───────────────┐    ┌────────────────────┐    ┌──────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────┐
│  Web Scraping │ →  │      MongoDB       │ →  │  ETL & Normalisation │ →  │        MotherDuck        │ →  │        Streamlit     │
│ APIs/Crawlers │    │ BDD NSql  (JSON)   │    │ Nettoyage & Enrich.  │    │ Data Warehouse étoile    │    │ Recherche & Analyses │
└───────────────┘    └────────────────────┘    └──────────────────────┘    └──────────────────────────┘    └──────────────────────┘
```

---

## 🌐 Sources de données

Quatre plateformes majeures ont été exploitées :

- **France Travail**  
  API officielle, OAuth2, scraping parallèle
- **APEC**  
  Selenium + BeautifulSoup, extraction structurée offline
- **JobTeaser**  
  Anti-bot, scraping React, filtrage précoce
- **Choisir le Service Public**  
  Scraping + extraction structurée assistée par LLM (Mistral)

Les données brutes sont stockées en **MongoDB Atlas** (NoSQL) au format **JSON**.

---

## 🗄️ Data Warehouse – MotherDuck

Le data warehouse repose sur **MotherDuck (DuckDB cloud)** avec :

- **Schéma en étoile**
- **Table de faits** : `f_offre`
- **Dimensions** : `d_date`, `d_contrat`, `d_localisation`, `h_region`

--- 

## 🤖 NLP & Machine Learning

### 🔎 Filtrage Data / Non-Data

Approche hybride :
- règles expertes (regex whitelist / blacklist)
- **TF-IDF + régression logistique**

Résultats :
- **F1-score : 0.978**
- **ROC-AUC : 0.996**
- **+67 %** d’offres data récupérées par rapport aux regex seules

---

### 🔍 Recherche sémantique
- Modèle : `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- Requêtes en **langage naturel**
- Similarité cosinus calculée **côté base** (DuckDB)

---

## 🖥️ Application Streamlit

Application **multi-pages** :
- Recherche sémantique, par mot clé et filtre
- Cartographie interactive (Folium + clustering)
- Tableaux de bord analytiques (Plotly)
- Ajout manuel d’offres et Chatbot LLM (Mistral) pour la structuration d’offres
- Clustering sémantique (UMAP + HDBSCAN)
- Graphe de co-occurrences des compétences

🔒 Connexion sécurisée à MotherDuck via token  

---



## Architecture du Projet 

```
RUCHE/
├── data/        # Bases DuckDB locales (backup et environnement de travail)
│   └── local.duckdb
│
├── docker/
│   └── Dockerfile.streamlit
│
├── documentation/      # Documentation et notices techniques
│   ├── notice_france_travail_scraper.md
│   ├── notice_moteur_recherche_semantique.md
│   ├── notice_TFIDF_ML_filtre_data_nondata.md
│   ├── Rapport.md 
│   ├── SISE NLP_Text Mining_Rapport_Groupe6_RUCHE.pdf
│   └── SISE NLP_Text Mining_Rapport_Groupe6_RUCHE.tex
│
├── dump/ # Stockage local des collections de la base NoSQL
│   ├── RUCHE_datalake/
│   └── prelude.json
│
├── etl/        # Pipeline ETL et traitements analytiques
│   ├── cleanX.py  # Fonctions de nettoyage et normalisation des données
│   ├── config_etl.py
│   ├── etl_mongo_mduck.py # Lancement principal d'ETL
│   ├── etl_utils.py
│   ├── etl_vectorization.py
│   ├── geolocation_enrichment.py # Enrichissement géographique (latitude / longitude via API)
│   └── tfidf_ml_data_filter.py  # Filtrage ML Data / Non-Data (TF-IDF + Logistic Regression)
│
├── mongodb/    # Alimentation et gestion de BDD NoSql MongoDB à partir de JSON du scrapping
│   ├── main_mongo.py
│   ├── mongodb_load_jobteaser.py
│   ├── mongodb_utils.py
│   └── reference_apec.py
│
├── ruche/
│    ├── __init__.py
│    └── db.py # Fonctions de connexion centralisé aux bases
│
├── scrapers/    # Scripts de collecte des offres d’emploi
│   ├── apec/
│   ├── francetravail/
│   ├── jobteaser/
│   └── service_public/
│
├── streamlit/   # Application web interactive Streamlit
│   ├── static/   # Ressources statiques (logos, images)
│   ├── .streamlit/ # contient le fichier config.toml pour le style de l'application
│   ├── 1_home_page.py
│   ├── 2_cartographie.py
│   ├── 3_visualisation.py
│   ├── 4_add_offers.py
│   ├── 5_clustering.py
│   ├── 6_graphe_competences.py
│   ├── 7_about.py
│   ├── app.py
│   └── config.py

│
├── .env.example # Exemple de .env 
├── .gitignore
├── .python-version # UV
├── docker-compose.yaml
├── pyproject.toml #UV       
├── README.md 
├── requirements_streamlit.txt # Requirements allegé pour Docker
└── requirements.txt 

```
--- 

## ⚙️ Installation & Lancement (Streamlit)

---

### 📋 Prérequis

- Docker ≥ 20.x
- Docker Compose
- Accès au dépôt du projet (Git ou Drive)

### 📥 Récupération du projet

Cloner le dépôt GitHub :

```bash
git clone https://github.com/RomainBuono/RUCHE.git
cd RUCHE
```

---

### 📁 Arborescence requise

Pour que l’application fonctionne correctement, les dossiers suivants doivent être présents :

```text
RUCHE/
├── data/                     # Base DuckDB (.duckdb)
├── streamlit/                # Application Streamlit
├── etl/                      # Fonctions ETL utilisées dans l’app
├── documentation/            # Rapport PDF chargé dans l’application
├── docker/
│   └── Dockerfile.streamlit  # Dockerfile de l’app Streamlit
├── ruche/                    # Méthodes de connexion à la base
├── requirements_streamlit.txt
├── docker-compose.yml
└── .env                      # Variables d’environnement (à créer)
```
- ⚠️ Le fichier .env doit être placé à la racine du projet, au même niveau que docker-compose.yml.
- ⚠️ Injecter les variables d’environnement via un fichier `.env`

### 🐳 Build de l’image Streamlit

Depuis la racine du projet :
```bash
docker build -f docker/Dockerfile.streamlit -t ruche-streamlit .
```

### ▶️ Lancement de l’application
```bash
docker compose up
```
L’application est ensuite accessible à l’adresse :
-> http://localhost:8501

Pour l' arrêt de l'application :
```bash
docker compose down
```
## ⚙️ Installation alternative (sans Docker) — recommandée pour les performances

Cette méthode correspond à une installation **locale ou avec connexions distantes**  
et constitue **l’alternative la plus rapide** à l’exécution via Docker.

L’installation via **Docker Compose** reste pertinente pour la reproductibilité,
mais peut entraîner des temps de chargement plus élevés pour l’application Streamlit.

---

### 📥 Récupération du projet

Cloner le dépôt GitHub :

```bash
git clone https://github.com/RomainBuono/RUCHE.git
cd RUCHE
```
---
### 🐍 Environnement Python

⚠️ Python **3.12** est requis (Python 3.13 non supporté).

Créer et activer un environnement virtuel (au choix) :

**Avec `venv`**
```bash
python3.12 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```
**Avec `conda`**
```bash
conda create -n ruche python=3.12
conda activate ruche
```
**Avec `uv`**
```bash
uv venv --python 3.12
source .venv/bin/activate
```
---
### 📦 Installation des dépendances
Installer les dépendances nécessaires à l’application Streamlit :
```bash
pip install -r requirements.txt
```
---
### 🔐 Variables d’environnement

Créer un fichier `.env` à la racine du projet contenant les variables suivantes 
```bash
### 🔐 Variables d’environnement

Créer un fichier `.env` à la racine du projet contenant les variables suivantes :

```env
# ---------- Connexion MongoDB distante
MONGO_URI=
MONGO_DATABASE=RUCHE_datalake
MISTRAL_API_KEY=

# ---------- Connexion DuckDB / MotherDuck
MOTHERDUCK_TOKEN=
MOTHERDUCK_DB=job_market_RUCHE   # La base doit exister dans MotherDuck (même vide)

# ---------- Connexion locale
DUCKDB_PATH=/data/local.duckdb

# ---------- Mode de connexion
# Valeurs possibles : offline (DuckDB) | online (MotherDuck)
CONNEXION_MODE=offline
```
⚠️ Le fichier `.env` **ne doit pas être versionné**.

---
### 🔄 Modes de connexion

L’application supporte deux modes de connexion, pilotés par la variable `CONNEXION_MODE` :

- **offline** : utilisation d’une base DuckDB locale  
- **online** : connexion à une base MotherDuck distante  

Le comportement de l’application Streamlit s’adapte automatiquement au mode sélectionné.

---
### ▶️ Lancement de l’application
Lancer l’application Streamlit :
```bash
cd streamlit
streamlit run app.py
```
L’application est accessible à l’adresse :
👉 http://localhost:8501

--- 

## 📚 Ressources associées

- 🎥 **Vidéo d'installation** : **[📺 Installation Demo Video](https://youtu.be/PWfyqenFNrk)**
- 🎥 **Vidéo de démonstration** : **[📺 Demonstration Video](https://youtu.be/gIqiBeyKKjI)**
- 📄 **Rapport académique (PDF)**  : [Projet NLP & Text Mining – Rapport RUCHE (Groupe 6)](/documentation/SISE%20NLP_Text%20Mining_Rapport_Groupe6_RUCHE.pdf)
- 📘 **Notice technique – Filtrage ML Data / Non-Data**  : [TF-IDF & Régression logistique](documentation/notice_TFIDF_ML_filtre_data_nondata.md)
- 📘 **Notice technique – Scraper France Travail**  : [API & Web Scraping France Travail](documentation/notice_france_travail_scraper.md)
- 📘 **Notice technique – Moteur de recherche sémantique**  : [Recherche vectorielle & similarité cosinus](documentation/notice_moteur_recherche_semantique.md)
- 📚 **API-Référentiel géographique français** : [API Géolocalisation](https://data.enseignementsup-recherche.gouv.fr/explore/dataset/fr-esr-referentiel-geographique/api/)




## 👥 Équipe

- Romain Buono
-  Yassine Cheniour
- Miléna Gordien-Piquet
- Anne-Camille Vial

#### 🎓 Master 2 SISE – Université Lyon 2
#### 👨‍🏫 Encadrant : M. Ricco Rakotomalala

---
