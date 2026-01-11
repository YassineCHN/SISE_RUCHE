import streamlit as st

st.set_page_config(layout="wide", page_title="A propos de RUCHE", page_icon=":honeybee:")
st.markdown(
    "<h1 style='text-align: center;'><b>  A propos ℹ️ </b></h1>", unsafe_allow_html=True
)
with st.container(horizontal=True):
    col1, col2 = st.columns(2)
    with col1:
        left = st.container(horizontal_alignment="center", border=True)
        left.markdown(
            "<h3 style='text-align: center;'><u>RUCHE (Réseau Unifié pour CHercher de l'Emploi)</u></h3>",
            unsafe_allow_html=True,
        )
        left.image(
            "./static/Logo3.png", width=350, caption="Le projet RUCHE constitue un système intégré d'acquisition, de structuration et d'analyse d'offres d'emploi dans les domaines de la data science et de l'intelligence artificielle. "
        )
    with col2:
        right = st.container(horizontal_alignment="center", border=True)
        right.markdown(
            "<h3 style='text-align: center;'><u> Architecture applicative : </u></h3>",
            unsafe_allow_html=True,
        )
        right.image(
            "./static/architecture.png",
            width="stretch",
            caption="Le système s'articule autour de quatre composantes principales: un ensemble de scrapers hétérogènes collectant des données depuis quatre plateformes majeures, un datalake NoSQL MongoDB assurant le stockage intermédiaire, un entrepôt de données MotherDuck structuré selon un modèle dimensionnel en étoile, et une application Streamlit multi-pages offrant des capacités de recherche sémantique et d'analyse géospatiale. L'architecture exploite des techniques avancées de NLP incluant la vectorisation sémantique par sentence-transformers, le filtrage par classification TF-IDF et régression logistique, ainsi que l'extraction assistée par modèles de langage. Le système agrège plusieurs milliers d'offres d'emploi géolocalisées et permet une exploration interactive du marché professionnel via des visualisations analytiques et cartographiques.",
        )
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #718096; font-size: 0.9rem;'>Powered by <strong>MotherDuck</strong> × <strong>Sentence Transformers</strong> | RUCHE Team © 2026</div>", unsafe_allow_html=True)
