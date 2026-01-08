import streamlit as st

# Main page content
st.markdown("# Home page 🎈")
st.sidebar.image("./static/Logo3.png")
st.sidebar.markdown("# Home page 🎈")

with st.container(horizontal=True):
    cl1, cl2, cl3, cl4 = st.columns(4)
    with cl1:
        st.image("./static/Logo3.png", width="content")

st.divider()

with st.container():
    st.header("Architecture applicative : ")
    st.image("./static/architecture.png", width="content")
