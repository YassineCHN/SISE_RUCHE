import os
import uuid
import datetime as dt
import streamlit as st
import pandas as pd
import duckdb
import sys
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from config import MOTHERDUCK_DATABASE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


load_dotenv(find_dotenv())
token = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_TOKEN = token

from etl.etl_utils import (
    normalize_company_name,
    normalize_education_level,
    serialize_list,
    force_list,
)

# -----------------------------
# Page config
# -----------------------------
st.markdown("# Ajout Offres 🆕")


# -----------------------------
# Connection helpers
# -----------------------------
def connect_duckdb_local() -> duckdb.DuckDBPyConnection:
    """
    Connexion DuckDB locale — DOIT pointer vers la base créée par l’ETL.
    """
    project_root = Path(__file__).resolve().parents[1]
    duckdb_path = project_root / "data" / "local.duckdb"

    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(duckdb_path))


def connect_motherduck() -> duckdb.DuckDBPyConnection:
    """
    Connexion MotherDuck — base distante unique.
    """
    return duckdb.connect(
        f"md:{MOTHERDUCK_DATABASE}?motherduck_token={MOTHERDUCK_TOKEN}"
    )


@st.cache_resource(show_spinner=False)
def get_connection(mode: str) -> duckdb.DuckDBPyConnection:
    """
    Cache Streamlit SAFE
    - clé = mode ('Local' ou 'MotherDuck')
    - une connexion par base
    """
    if mode == "MotherDuck":
        return connect_motherduck()
    return connect_duckdb_local()


# -----------------------------
# DB read helpers
# -----------------------------
def fetch_df(con: duckdb.DuckDBPyConnection, query: str) -> pd.DataFrame:
    return con.execute(query).fetchdf()


def get_or_create_date_id(con: duckdb.DuckDBPyConnection, d: dt.date) -> int:
    """
    FK d_date : on récupère l'id_date si la date existe,
    sinon on insère une ligne (contrôlé) et on retourne le nouvel id.
    """
    # d_date(date_complete DATE, id_date INTEGER PK, ...)
    existing = con.execute(
        "SELECT id_date FROM d_date WHERE date_complete = ? LIMIT 1",
        [d],
    ).fetchone()

    if existing:
        return int(existing[0])

    # Calculs type ETL (simple, suffisant et cohérent)
    # Note: les noms mois/jour seront en anglais si on fait strftime sans locale ;
    # c'est OK analytiquement, ou tu peux imposer une locale plus tard.
    next_id = con.execute(
        "SELECT COALESCE(MAX(id_date), 0) + 1 FROM d_date"
    ).fetchone()[0]
    next_id = int(next_id)

    trimestre = (d.month - 1) // 3 + 1
    semaine = int(d.isocalendar().week)
    jour_annee = int(d.timetuple().tm_yday)

    con.execute(
        """
        INSERT INTO d_date (
            id_date, date_complete, jour, mois, annee, trimestre,
            nom_mois, nom_jour, semaine, jour_annee
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            next_id,
            d,
            d.day,
            d.month,
            d.year,
            trimestre,
            d.strftime("%B"),
            d.strftime("%A"),
            semaine,
            jour_annee,
        ],
    )
    return next_id


# -----------------------------
# UI / Form
# -----------------------------
st.caption("Choisis la base cible (local ou MotherDuck).")

mode = st.radio("Base", ["Local DuckDB", "MotherDuck"], horizontal=True)
con = get_connection("MotherDuck" if mode == "MotherDuck" else "Local DuckDB")
st.session_state["duckdb_connection"] = con

# Charger dimensions
with st.spinner("Chargement des dimensions..."):
    df_contrat = fetch_df(
        con, "SELECT id_contrat, type_contrat FROM d_contrat ORDER BY id_contrat"
    )
    df_ville = fetch_df(
        con, "SELECT id_ville, ville FROM d_localisation ORDER BY ville"
    )

# Mappings
contract_label_to_id = dict(zip(df_contrat["type_contrat"], df_contrat["id_contrat"]))
ville_label_to_id = dict(zip(df_ville["ville"], df_ville["id_ville"]))

# Référentiels UI (simples)
EDU_OPTIONS = ["UNKNOWN", "AUCUN_PREREQUIS", "BAC", "BAC+2", "BAC+3", "BAC+5"]

tab1, tab2, tab3 = st.tabs(["Formulaire", "Scraping", "LLM"])

with tab1:
    st.subheader("Nouvelle offre (insertion en base)")

    with st.form("add_offer_form", enter_to_submit=False):
        st.markdown("### Champs importants ❗: ")
        colA, colB = st.columns(2)

        with colA:
            title = st.text_input("Titre *", placeholder="Ex: Data Analyst (H/F)")
            source_url = st.text_input(
                "URL source (optionnel)", placeholder="https://..."
            )
            company_name = st.text_input("Entreprise", placeholder="Ex: Acme SAS")

            description = st.text_area("Description *", height=180)

        with colB:

            contract_type = st.selectbox(
                "Type de contrat *",
                options=list(contract_label_to_id.keys()),
                index=(
                    list(contract_label_to_id.keys()).index("AUTRE")
                    if "AUTRE" in contract_label_to_id
                    else 0
                ),
            )
            DUREE_AUTORISEE = contract_type not in ["CDI"]

            duree_contrat_mois = st.number_input(
                "Durée contrat (mois)",
                min_value=0,
                max_value=120,
                value=0,
                disabled=not DUREE_AUTORISEE,
            )
            if not DUREE_AUTORISEE:
                duree_contrat_mois = None
            elif duree_contrat_mois == 0:
                duree_contrat_mois = None

            ville = st.selectbox(
                "Ville (dimension) *",
                options=list(ville_label_to_id.keys()),
                index=(
                    list(ville_label_to_id.keys()).index("UNKNOWN")
                    if "UNKNOWN" in ville_label_to_id
                    else 0
                ),
            )

            is_teletravail = st.checkbox("Télétravail", value=False)
            SALARY_OPTIONS = [
                "< 25k€",
                "25k€ - 30k€",
                "30k€ - 35k€",
                "35k€ - 40k€",
                "40k€ - 45k€",
                "45k€ - 50k€",
                "50k€ - 60k€",
                "60k€ - 70k€",
                "70k€ - 80k€",
                "80k€ - 100k€",
                "> 100k€",
                "A négocier",
            ]
            salaire = st.selectbox(
                "Salaire",
                options=[""] + SALARY_OPTIONS,
                index=0,
                help="Fourchette brute annuelle",
            )
            salaire = salaire if salaire != "" else None

            # --- Start date (wizard + UI améliorée) ---
            months = [
                "Janvier",
                "Février",
                "Mars",
                "Avril",
                "Mai",
                "Juin",
                "Juillet",
                "Août",
                "Septembre",
                "Octobre",
                "Novembre",
                "Décembre",
            ]
            current_year = dt.date.today().year
            years = list(range(current_year, current_year + 3))
            # Initialisation état
            if "start_mode" not in st.session_state:
                st.session_state.start_mode = "Dès que possible"
            if "start_show_month_year" not in st.session_state:
                st.session_state.start_show_month_year = False
            if "start_month" not in st.session_state:
                st.session_state.start_month = months[0]
            if "start_year" not in st.session_state:
                st.session_state.start_year = years[0]
            # Ligne principale
            col_radio, col_btn, col_month, col_year = st.columns([3, 1, 2, 2])
            with col_radio:
                st.radio(
                    "Date de début",
                    options=["Dès que possible", "Choisir un mois"],
                    horizontal=True,
                    key="start_mode",
                )
            with col_btn:
                col_btn.space("small")
                apply_start = st.form_submit_button("Choix date ➡️")
            if apply_start:
                st.session_state.start_show_month_year = (
                    st.session_state.start_mode == "Choisir un mois"
                )
            # Champs Mois / Année affichés à droite
            if st.session_state.start_show_month_year:
                with col_month:
                    st.selectbox("Mois", months, key="start_month")
                with col_year:
                    st.selectbox("Année", years, key="start_year")
            # Valeur finale stockée
            start_date = (
                "Dès que possible"
                if st.session_state.start_mode == "Dès que possible"
                else f"{st.session_state.start_month} {st.session_state.start_year}"
            )
        HARD_SKILLS_REF = [
            "Python",
            "SQL",
            "Power BI",
            "Tableau",
            "Airflow",
            "Spark",
            "Docker",
            "Git",
            "Machine Learning",
            "Deep Learning",
        ]
        SOFT_SKILLS_REF = [
            "Communication",
            "Autonomie",
            "Esprit d’analyse",
            "Travail en équipe",
            "Rigueur",
            "Curiosité",
        ]
        LANGUAGES_REF = ["Français", "Anglais", "Espagnol", "Allemand"]
        st.divider()
        st.markdown("### Compétences 🧰:")

        col1, col2, col3 = st.columns(3)
        with col1:
            hard_skills = st.multiselect(
                "Hard skills",
                options=HARD_SKILLS_REF,
                default=[],
                accept_new_options=True,
            )
        with col2:
            soft_skills = st.multiselect(
                "Soft skills",
                options=SOFT_SKILLS_REF,
                default=[],
                accept_new_options=True,
            )
        with col3:
            langages = st.multiselect(
                "Langages",
                options=LANGUAGES_REF,
                default=[],
                accept_new_options=True,
            )
        with st.expander("➕ Champs optionnels"):
            col1, col2 = st.columns(2)
            with col1:
                company_description = st.text_area(
                    "Description de l'entreprise", height=100
                )

                job_grade = st.text_input(
                    "Job grade", placeholder="Ex: Junior / Senior / Cadre"
                )
                job_function = st.text_input(
                    "Job function", placeholder="Ex: Data / BI / Analytics"
                )
            with col2:
                today = dt.date.today()
                MIN_DATE = dt.date(2025, 1, 1)
                MAX_DATE = today + dt.timedelta(days=3)
                publication_date = st.date_input(
                    "Date de publication",
                    value=today,
                    min_value=MIN_DATE,
                    max_value=MAX_DATE,
                )
                if publication_date > today:
                    st.warning("⚠️ Date de publication dans le futur.")
                education_level_raw = st.selectbox(
                    "Niveau d'études", options=EDU_OPTIONS, index=0
                )
                experience_required = st.text_input(
                    "Expérience spécifique requise",
                    placeholder="Ex: 2 ans sur un poste similaire",
                )
                experience_years = st.slider(
                    "Années d'expérience requises", min_value=0, max_value=15, value=0
                )

        submit = st.form_submit_button("Ajouter")
        if submit:

            # Validations
            errors = []
            if not title.strip():
                errors.append("Le champ **Titre** est obligatoire.")
            if not description.strip():
                errors.append("Le champ **Description** est obligatoire.")
            if contract_type not in contract_label_to_id:
                errors.append("Type de contrat invalide.")
            if ville not in ville_label_to_id:
                errors.append("Ville invalide (dimension).")
            if errors:
                for e in errors:
                    st.error(e)
                    st.stop()

            # Normalisations ETL
            company_name_norm = normalize_company_name(company_name)
            education_level_norm = normalize_education_level(education_level_raw)

            # Multi-valeurs → listes → JSON TEXT
            hard_list = list(dict.fromkeys(hard_skills))
            soft_list = list(dict.fromkeys(soft_skills))
            lang_list = list(dict.fromkeys(langages))

            hard_json = serialize_list(force_list(hard_list))
            soft_json = serialize_list(force_list(soft_list))
            lang_json = serialize_list(force_list(lang_list))

            # FK
            id_contrat = int(contract_label_to_id[contract_type])
            id_ville = int(ville_label_to_id[ville])
            id_date_publication = get_or_create_date_id(con, publication_date)

            # ID + scraped_at
            job_id = str(uuid.uuid4())
            scraped_at = dt.date.today()
            source_platform = "STREAMLIT"

            # Insert fact
            try:
                con.execute(
                    """
                INSERT INTO f_offre (
                    job_id, source_platform, source_url, scraped_at,
                    title, description,
                    company_name, company_description,
                    nb_annees_experience, experience_required,
                    id_ville, id_contrat, duree_contrat_mois,
                    id_date_publication, start_date,
                    is_teletravail, salaire,
                    hard_skills, soft_skills, langages,
                    education_level, job_function, job_grade,
                    None, None
                )
                VALUES (
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?
                )
                """,
                    [
                        job_id,
                        source_platform,
                        source_url or None,
                        scraped_at,
                        title,
                        description,
                        company_name_norm,
                        company_description or None,
                        int(experience_years) if experience_years is not None else None,
                        experience_required or None,
                        id_ville,
                        id_contrat,
                        duree_contrat_mois,
                        id_date_publication,
                        start_date or None,
                        bool(is_teletravail),
                        salaire or None,
                        hard_json,
                        soft_json,
                        lang_json,
                        education_level_norm,
                        job_function or None,
                        job_grade or None,
                        False,
                        0.0,
                    ],
                )
                con.execute("CHECKPOINT")
                st.success("✅ Offre ajoutée avec succès !")
                st.info(f"job_id = {job_id}")
                # Petit check d’affichage
                inserted = con.execute(
                    "SELECT * FROM f_offre WHERE job_id = ?",
                    [job_id],
                ).fetchdf()
                st.dataframe(inserted, width="stretch")
            except Exception as e:
                st.error(f"❌ Insertion échouée: {e}")

with tab2:
    st.subheader("Ajout via scraping (à implémenter)")
    st.caption("Placeholder: tu pourras ici brancher un extracteur par URL.")
    url = st.text_input("Entrez une URL à ajouter", value="")
    if st.button("Valider", icon="✅"):
        if not url.strip():
            st.warning("L'URL fournie est vide.")
        else:
            st.write("URL reçue :", url)

with tab3:
    st.subheader("LLM (à implémenter)")
    st.caption("Placeholder: enrichissement / reformulation / extraction structurée.")
