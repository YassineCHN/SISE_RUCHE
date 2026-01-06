import streamlit as st

# Main page content
st.markdown("# Home page 🎈")
st.sidebar.image("./static/Logo3.png")
st.sidebar.image("./static/Logo4_bis.png")
st.sidebar.markdown("# Home page 🎈")

with st.container(horizontal=True):
    cl1, cl2, cl3, cl4 = st.columns(4)
    with cl1:
        st.image("./static/Logo3.png", width="content")
    with cl2:
        st.image("./static/Logo4_bis.png", width="content")
    with cl3:
        st.image("./static/Logo_bis.png", width="content")
    with cl4:
        st.image("./static/Logo2_bis.png", width="content")
