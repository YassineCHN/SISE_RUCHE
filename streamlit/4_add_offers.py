import streamlit as st

st.markdown("# Ajout Offres 🆕")

# la sidebar n'apparait pas parce qu'il n'y a pas le st.sidebar sur cette page
flex = st.container(horizontal=True, vertical_alignment="bottom")
URL = flex.text_input(label="Entrez un URL à ajouter", value="httpsFZ")
if flex.button(label="Valider", icon="✅"):
    if URL == "":
        st.write("L'URL fourni est vide")
    else:
        st.write(URL)
