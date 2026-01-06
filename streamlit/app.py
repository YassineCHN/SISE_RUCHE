import streamlit as st
import numpy as np
import pandas as pd
import time

st.set_page_config(layout="wide")

# Define the pages
home_page = st.Page("1_home_page.py", title="Home Page", icon="🏠")
cartographie = st.Page("2_cartographie.py", title="Cartographie", icon="🌍")
visualisation = st.Page("3_visualisation.py", title="Visualisation", icon="📊")
ajout = st.Page("4_add_offers.py", title="Ajout Offres", icon="🆕")
# Set up navigation
pg = st.navigation([home_page, cartographie, visualisation, ajout], position="top")

# Run the selected page
pg.run()
