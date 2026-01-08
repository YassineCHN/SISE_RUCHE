import streamlit as st
import datetime

st.markdown("# Ajout Offres 🆕")

# la sidebar n'apparait pas parce qu'il n'y a pas le st.sidebar sur cette page

etudes = ["ajouter", "les", "vrais", "champs", "de la base"]
list_langues = [
    "Aucune",
    "Français",
    "Anglais",
    "Espagnol",
    "Allemand",
    "Japonais",
    "Chinois",
    "Arabe",
    "Russe",
    "Coréen",
    "Grec",
    "Malgache",
]
list_soft_skills = [
    "Récupérer",
    "les",
    "skills",
    "de",
    "la",
    "base",
    "pas tel quel, faire une liste restreinte",
]
list_hard_skills = ["Récupérer", "les", "skills", "de", "la", "base"]

tab1, tab2, tab3 = st.tabs(["Formulaire", "Scraping", "LLM"])

with tab1:
    with st.form("add_offer_form", enter_to_submit=False):
        st.empty()
        description_field = st.text_area("Description")
        company = st.text_input("Entreprise")
        company_description = st.text_input("Description de l'entreprise")
        annee_requise = st.slider(
            "Années d'expérience requise", min_value=0, max_value=10
        )
        experience_requise = st.text_input("Expérience spécifique requise?")
        # ville=
        # region=
        # contrat_type
        date_pub = st.date_input(
            "Date de publication", value="today", format="DD/MM/YYYY"
        )  # empeche de mettre une date futur à aujourd'hui
        teletravail = st.checkbox("Télétravail")
        salary = st.text_input("Salaire")
        softskills = st.multiselect(
            "Soft skills",
            options=list_soft_skills,
            accept_new_options=True,
        )
        hardskills = st.multiselect(
            "Hards skills",
            options=list_hard_skills,
            accept_new_options=True,
        )
        langues = st.multiselect(
            "Langues nécessaires",
            options=list_langues,
            accept_new_options=True,
        )  # si vide mettre "Aucune"
        etudes = st.selectbox("Niveau d'études requis", options=etudes)
        # fonctio=
        # sector=
        # job_grade=
        submit = st.form_submit_button("Ajouter")


# if submit:
#    source_platform="streamlit"
#    description=description_field
#    company_name=company
#    company_description (if company in db.company_name: company_description=db.company_description )
#    nb_annees_experience=annee_requise
#    experience_required=experience_requise
#    id_ville=
#    id_region=
#    id_contrat_type=
#    salaire=salary
#    is_teletravail=teletravail
#    date_publication=date_pub
#    soft_skills=softskills
#    hard_skills=hardskills
#    langages=langues
#    education_level=etudes

#    job_function=
#    sector=
#    job_grade=


with tab2:
    flex = st.container(horizontal=True, vertical_alignment="bottom")
    URL = flex.text_input(label="Entrez un URL à ajouter", value="httpsFZ")
    if flex.button(label="Valider", icon="✅"):
        if URL == "":
            st.write("L'URL fourni est vide")
        else:
            st.write(URL)
with tab3:
    st.empty()
