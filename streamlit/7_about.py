import streamlit as st
import base64
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(layout="wide", page_title="À propos de RUCHE", page_icon="🐝")

# ============================================================
# CHEMINS ROBUSTES (indépendants du dossier où tu lances streamlit)
# ============================================================
CURRENT_DIR = Path(__file__).resolve().parent  # ex: streamlit_app/
PROJECT_ROOT = CURRENT_DIR.parent  # racine du repo

LOGO_PATH = CURRENT_DIR / "static" / "Logo3.png"
ARCHI_PATH = CURRENT_DIR / "static" / "architecture.png"
PDF_PATH = (
    PROJECT_ROOT / "documentation" / "SISE NLP_Text Mining_Rapport_Groupe6_RUCHE.pdf"
)

# ============================================================
# TITRE
# ============================================================
st.markdown(
    """
    <h1 style="text-align: center; margin-bottom: 0.2rem;"><b>À propos ℹ️</b></h1>
    <p style="text-align: center; color: #6b7280; font-size: 1.05rem; margin-top: 0;">
        RUCHE — Réseau Unifié pour la Recherche d’Emploi (Data & IA)
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SECTION 1 : PROJET + ARCHITECTURE
# ============================================================
col1, col2 = st.columns(2, gap="large")

with col1:
    left = st.container(border=True)

    left.markdown(
        "<h3 style='text-align: center;'><u>🎯 Le projet RUCHE</u></h3>",
        unsafe_allow_html=True,
    )

    # --- Image centrée ---
    c1, c2, c3 = left.columns([2, 3, 2])
    with c2:
        st.image(str(LOGO_PATH), width=500)

    # --- Description SOUS l’image ---
    left.markdown(
        """
        <p style="text-align: justify; font-size: 1rem; margin-top: 1rem;">
            <b>RUCHE</b> est un système intégré d’acquisition, de structuration et d’analyse
            d’offres d’emploi dans les domaines de la <b>data science</b> et de l’
            <b>intelligence artificielle</b>.
            L’objectif est de centraliser des sources hétérogènes et de fournir une exploration
            fiable et analytique du marché de l’emploi.
        </p>
        """,
        unsafe_allow_html=True,
    )
with col2:
    right = st.container(border=True)

    right.markdown(
        "<h3 style='text-align: center;'><u>🏗️ Architecture applicative</u></h3>",
        unsafe_allow_html=True,
    )

    # --- Image centrée ---
    img_col1, img_col2, img_col3 = right.columns([1, 2, 1])
    with img_col2:
        right.image(str(ARCHI_PATH), use_container_width=True)

    # --- Description SOUS l’image ---
    right.markdown(
        """
        <p style="text-align: justify; font-size: 1rem; margin-top: 1rem;">
            Le système s’articule autour de <b>quatre composantes principales</b> :
            <br>
            <b>-</b> Scrapers multi-sources (plateformes d’emploi)<br>
            <b>-</b> BDD NoSQL <b>MongoDB</b> (stockage brut/intermédiaire)<br>
            <b>-</b> Entrepôt <b>MotherDuck</b> (modèle dimensionnel en étoile)<br>
            <b>-</b> Application <b>Streamlit</b> multi-pages (analyse & visualisation)<br>
            <br>
            L’enrichissement s’appuie sur des techniques de <b>NLP</b> :
            <i>Sentence Transformers</i>, <i>TF-IDF</i> et modèles de langage.
        </p>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# SECTION 2 : CHIFFRES CLÉS + STACK + ÉQUIPE
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)

colA, colB = st.columns(2, gap="large")
with colA:
    st.subheader("🧰 Technologies principales")
    st.markdown(
        """
        - **Python** (scraping, ETL, NLP)
        - **MongoDB** (NoSql BDD)
        - **DuckDB / MotherDuck** (entrepôt analytique)
        - **Streamlit** (application multi-pages)
        - **Sentence Transformers** (vectorisation sémantique)
        - **TF-IDF + Régression logistique** (filtrage)
        - ... et bien d’autres ! 🚀
        """
    )

with colB:
    st.subheader("👥 Équipe")
    st.markdown(
        """
        - Romain BUONO
        - Yassine CHENIOUR
        - Anne-Camille  VIAL
        - Milena GORDIEN PIQUET
        """
    )

# ============================================================
# SECTION 3 : RAPPORT PDF DANS “À propos”
# ============================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <h2 style="text-align: center; margin-bottom: 0.2rem;">📄 Rapport du projet</h2>
    <p style="text-align: center; color: #6b7280; font-size: 1rem; margin-top: 0;">
        Rapport académique (Text Mining & NLP) — Projet RUCHE
    </p>
    """,
    unsafe_allow_html=True,
)

if not PDF_PATH.exists():
    st.warning("⚠️ Le rapport PDF n’est pas trouvé.")
    st.caption(f"Chemin recherché : {PDF_PATH}")
else:
    pdf_bytes = PDF_PATH.read_bytes()
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    st.download_button(
        "⬇️ Télécharger le rapport (PDF)",
        data=pdf_bytes,
        file_name=PDF_PATH.name,
        mime="application/pdf",
    )

    pdf_display = f"""
    <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        width="100%"
        height="900"
        style="border: 1px solid #e5e7eb; border-radius: 12px; margin-top: 12px;"
        type="application/pdf">
    </iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; color: #718096; font-size: 0.9rem;">
        Powered by <strong>MotherDuck</strong> × <strong>Sentence Transformers</strong> |
        RUCHE Team © 2026
    </div>
    """,
    unsafe_allow_html=True,
)
