import streamlit as st

st.markdown("# Visualisation 📊")
st.divider()
st.sidebar.markdown("# 📊 Visualisation ")

col1, col2, col3 = st.columns(3)

col1.metric("Nombres d'offres", "X", border=True)
col2.metric("Salaire moyen", "X €", border=True)
col3.metric("Autres", "X", border=True)
st.divider()

column1, column2 = st.columns(2)
with column1:
    with st.container(border=True):
        st.write("Graphique des compétences demandées")
with column2:
    with st.container(border=True):
        st.write("Clustering d'offre")


with st.container(border=False, horizontal=True, horizontal_alignment="center"):
    st.markdown(
        "<p style='text-align: center;'>Date de dernière Mise à jour de la BDD : DD//MM//YYYY</p>",
        unsafe_allow_html=True,
    )
