import streamlit as st

# Main page content
st.markdown("# Home page 🎈")
st.sidebar.image("./static/Logo3.png")
st.sidebar.markdown("# Home page 🎈")

with st.container(horizontal=True):
    col1, col2 = st.columns(2)
    with col1:
        st.header("RUCHE")  # le passer en markdown
        st.image("./static/Logo3.png", width="content")
        st.write("le projet RUCHE vise à ...")
    with col2:
        st.header("Architecture applicative : ")
        st.image("./static/architecture.png", width="content")


st.divider()

with st.container():
    st.header("Moteur de recherche")
