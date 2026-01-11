import streamlit as st

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
            "./static/Logo3.png", width=350, caption="le projet RUCHE vise à ..."
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
            caption="l'application s'appuie sur ...",
        )
