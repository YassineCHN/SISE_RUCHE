import os
import duckdb
import uuid
import datetime as dt
import streamlit as st
import pandas as pd
from ruche.db import get_connection
import sys
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import nltk
from nltk.corpus import stopwords
import re
import unicodedata
import json
from mistralai import Mistral

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


load_dotenv(find_dotenv())
token = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_TOKEN = token

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


from etl.etl_utils import (
    normalize_company_name,
    normalize_education_level,
    serialize_list,
    force_list,
)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

FRENCH_STOPWORDS = stopwords.words("french")
# -----------------------------
# Page config
# -----------------------------


def call_llm_format_offer(raw_text: str) -> dict:
    """
    Analyse une description brute d'offre et retourne un dict structuré
    via le client Mistral officiel.
    """
    if not MISTRAL_API_KEY:
        raise RuntimeError(
            "MISTRAL_API_KEY manquant dans les variables d'environnement"
        )

    client = Mistral(api_key=MISTRAL_API_KEY)

    prompt = f"""
Tu es un assistant chargé d'extraire et structurer une offre d'emploi.

Retourne UNIQUEMENT un JSON valide avec les champs suivants :
- title
- description
- company_name
- type_contrat
- ville
- salaire
- hard_skills
- soft_skills
- langages

Texte de l'offre :
\"\"\"
{raw_text}
\"\"\"
"""

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},  # ⭐ IMPORTANT
    )

    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Réponse LLM non JSON valide.\n\nRAW:\n{content}") from e


st.markdown("# Ajout Offres 🆕")
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
</style>
""",
    unsafe_allow_html=True,
)
CURRENT_DIR = Path(__file__).resolve().parent
LOGO_PATH = CURRENT_DIR / "static" / "Logo3.png"

st.markdown("## 🧠 Assistance IA (optionnelle)")
with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 🐝 RUCHE")
        st.image(str(LOGO_PATH), width=140)

    st.markdown("## 🆕 Ajout d’une offre")
    st.caption(
        "Cette page permet d’ajouter manuellement une offre d’emploi "
        "dans la base analytique RUCHE."
    )

    st.divider()

    st.markdown("### 🧠 Assistance IA")
    st.markdown(
        "- Collez une offre brute\n"
        "- L’IA extrait automatiquement les champs clés\n"
        "- Vous pouvez modifier avant insertion"
    )

    st.divider()

    st.markdown("### ⚠️ Bonnes pratiques")
    st.markdown(
        "- Vérifiez le **titre** et la **description**\n"
        "- Sélectionnez la **ville exacte**\n"
        "- Évitez les doublons"
    )
raw_offer = st.text_area(
    "Collez ici une offre brute (copiée depuis une annonce)",
    height=200,
    placeholder="Collez ici le texte complet de l'offre...",
)

analyze_llm = st.button("Analyser avec l’IA")
if analyze_llm:
    if not raw_offer.strip():
        st.warning("Veuillez coller une offre avant de lancer l'analyse.")
    else:
        with st.spinner("Analyse de l'offre par l'IA..."):
            try:
                llm_data = call_llm_format_offer(raw_offer)
                st.session_state["llm_offer"] = llm_data
                st.success("✅ Offre analysée. Formulaire prérempli ci-dessous.")
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse IA : {e}")


# -----------------------------
# Connection helpers
# -----------------------------
def merge_options_with_defaults(options: list[str], defaults: list[str]) -> list[str]:
    """
    Garantit que toutes les valeurs par défaut sont présentes dans les options
    (requis par st.multiselect).
    """
    merged = list(options)
    for d in defaults:
        if d not in merged:
            merged.append(d)
    return merged


def map_llm_ville(llm_ville: str, available: list[str]) -> str | None:
    if not llm_ville:
        return None
    v = normalize_text(llm_ville)
    for a in available:
        if normalize_text(a) == v:
            return a
    return None


def map_llm_contract(llm_value: str, available: list[str]) -> str | None:
    if not llm_value:
        return None
    v = llm_value.strip().upper()
    for a in available:
        if v in a.upper():
            return a
    return None


def normalize_llm_list(value):
    """
    Normalise une valeur issue du LLM vers une liste de strings.
    Accepte :
    - list[str]
    - "a;b;c"
    - "none" / None
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    if isinstance(value, str):
        if value.lower() in {"none", "unknown", ""}:
            return []
        return [v.strip() for v in value.split(";") if v.strip()]

    return []


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


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def is_exact_duplicate(title, description, candidates):
    title_norm = normalize_text(title)
    desc_norm = normalize_text(description)

    for _, row in candidates.iterrows():
        if (
            normalize_text(row["title"]) == title_norm
            and normalize_text(row["description"]) == desc_norm
        ):
            return True, 1.0, row["job_id"]

    return False, 0.0, None


def detect_duplicate_streamlit(
    con,
    title: str,
    description: str,
    threshold: float = 0.85,
    max_candidates: int = 500,
):
    """
    Détection légère de doublons pour une insertion Streamlit.
    Compare la nouvelle offre aux offres existantes (title + description).
    """

    # 1️⃣ Récupérer un sous-ensemble pertinent
    candidates = con.execute(
        """
        SELECT job_id, title, description
        FROM f_offre
        WHERE title IS NOT NULL
        AND title != ''
        ORDER BY scraped_at DESC NULLS LAST
        LIMIT ?
        """,
        [max_candidates],
    ).fetchdf()

    if candidates.empty:
        return False, 0.0, None

    is_dup_exact, score_exact, dup_id_exact = is_exact_duplicate(
        title, description, candidates
    )
    if is_dup_exact:
        return True, score_exact, dup_id_exact
    # 2️⃣ Corpus
    new_text = f"{title} {description}"
    corpus = [new_text] + (
        candidates["title"].fillna("") + " " + candidates["description"].fillna("")
    ).tolist()

    # 3️⃣ Vectorisation
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words=FRENCH_STOPWORDS,  # ✅ LISTE
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents="unicode",
    )

    tfidf = vectorizer.fit_transform(corpus)

    # 4️⃣ Similarité
    similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    best_score = similarities.max()
    best_idx = similarities.argmax()

    if best_score >= threshold:
        duplicate_job_id = candidates.iloc[best_idx]["job_id"]
        return True, float(best_score), duplicate_job_id

    return False, float(best_score), None


# -----------------------------
# UI / Form
# -----------------------------
con = get_connection()
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

st.subheader("Nouvelle offre (insertion en base)")
llm_offer = st.session_state.get("llm_offer", {})

with st.form("add_offer_form", enter_to_submit=False):
    st.markdown("### Champs importants ❗: ")
    colA, colB = st.columns(2)

    with colA:
        title = st.text_input(
            "Titre *",
            value=llm_offer.get("title", ""),
            placeholder="Ex: Data Analyst (H/F)",
        )
        source_url = st.text_input("URL source (optionnel)", placeholder="https://...")
        description = st.text_area(
            "Description *",
            value=llm_offer.get("description", ""),
            height=180,
        )
        company_name = st.text_input(
            "Entreprise *",
            value=llm_offer.get("company_name", ""),
        )

    with colB:

        llm_contract = llm_offer.get("type_contrat")
        mapped_contract = map_llm_contract(
            llm_contract,
            list(contract_label_to_id.keys()),
        )

        contract_options = list(contract_label_to_id.keys())
        contract_index = (
            contract_options.index(mapped_contract)
            if mapped_contract in contract_options
            else (contract_options.index("AUTRE") if "AUTRE" in contract_options else 0)
        )
        contract_type = st.selectbox(
            "Type de contrat *",
            options=contract_options,
            index=contract_index,
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

        llm_ville = llm_offer.get("ville")
        mapped_ville = map_llm_ville(
            llm_ville,
            list(ville_label_to_id.keys()),
        )
        ville_options = list(ville_label_to_id.keys())
        ville_index = (
            ville_options.index(mapped_ville)
            if mapped_ville in ville_options
            else (ville_options.index("UNKNOWN") if "UNKNOWN" in ville_options else 0)
        )
        ville = st.selectbox(
            "Ville (dimension) *",
            options=ville_options,
            index=ville_index,
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
        llm_salary = llm_offer.get("salaire")
        salary_index = 0
        if llm_salary in SALARY_OPTIONS:
            salary_index = SALARY_OPTIONS.index(llm_salary) + 1  # +1 car "" en 0

        salaire = st.selectbox(
            "Salaire",
            options=[""] + SALARY_OPTIONS,
            index=salary_index,
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
        llm_hard = llm_offer.get("hard_skills")
        hard_default = normalize_llm_list(llm_hard)
        hard_options = merge_options_with_defaults(
            HARD_SKILLS_REF,
            hard_default,
        )
        hard_skills = st.multiselect(
            "Hard skills",
            options=hard_options,
            default=hard_default,
            accept_new_options=True,
        )
    with col2:
        llm_soft = llm_offer.get("soft_skills")
        soft_default = normalize_llm_list(llm_soft)
        soft_options = merge_options_with_defaults(
            SOFT_SKILLS_REF,
            soft_default,
        )
        soft_skills = st.multiselect(
            "Soft skills",
            options=soft_options,
            default=soft_default,
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
        if not ville.strip():
            errors.append("Le champ **Entreprise** est obligatoire")
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

        BLOCKING_THRESHOLD = 0.9

        is_dup, sim_score, dup_job_id = detect_duplicate_streamlit(
            con,
            title=title,
            description=description,
            threshold=BLOCKING_THRESHOLD,
        )
        # 🚫 Blocage si doublon quasi certain
        if is_dup and sim_score >= BLOCKING_THRESHOLD:
            st.error(
                f"❌ Insertion bloquée : cette offre est quasi identique à une offre existante "
                f"(similarité = {sim_score:.2f})."
            )
            st.info(f"Doublon détecté avec job_id : {dup_job_id}")
            st.stop()  # ⛔ STOP AVANT INSERT
        is_duplicate_db = bool(is_dup)
        similarity_score_db = float(sim_score) if is_dup else None

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
                is_duplicate, similarity_score
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
                    is_duplicate_db,
                    similarity_score_db,
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

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: #718096; font-size: 0.9rem;'>Powered by <strong>MotherDuck</strong> × <strong>Sentence Transformers</strong> | RUCHE Team © 2026</div>",
    unsafe_allow_html=True,
)
