import streamlit as st

# Main page content
st.markdown(
    "<h1 style='text-align: center;'><b>  Home page 🏠</b></h1>", unsafe_allow_html=True
)
st.sidebar.image("./static/Logo3.png", width=150)
st.sidebar.markdown("# Home page")

with st.container(horizontal=True):
    col1, col2 = st.columns(2)
    with col1:
        left = st.container(horizontal_alignment="center", border=True)
        left.markdown(
            "<h2 style='text-align: center;'><u>RUCHE (Réseau Unifié pour CHercher de l'Emploi)</u></h2>",
            unsafe_allow_html=True,
        )
        left.image(
            "./static/Logo3.png", width=350, caption="le projet RUCHE vise à ..."
        )
    with col2:
        right = st.container(horizontal_alignment="center", border=True)
        right.markdown(
            "<h2 style='text-align: center;'><u> Architecture applicative : </u></h2>",
            unsafe_allow_html=True,
        )
        right.image(
            "./static/architecture.png",
            width="stretch",
            caption="l'application s'appuie sur ...",
        )

st.divider()

with st.container():
    st.header("Moteur de recherche")
