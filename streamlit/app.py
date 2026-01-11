import streamlit as st
import numpy as np
import pandas as pd
import time
import atexit
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
st.set_page_config(layout="wide")

# Define the pages
home_page = st.Page("1_home_page.py", title="Home Page", icon="🏠")
cartographie = st.Page("2_cartographie.py", title="Cartographie", icon="🌍")
visualisation = st.Page("3_visualisation.py", title="Visualisation", icon="📊")
ajout = st.Page("4_add_offers.py", title="Ajout Offres", icon="🆕")
clustering = st.Page("5_Clustering.py", title="Clustering Offres", icon="🎯")
graphe_comp = st.Page("6_Graphe_competences.py", title="Graphe compétences", icon="🧰")
about = st.Page("7_about.py", title="A propos", icon="ℹ️")
# Set up navigation
pg = st.navigation(
    [
        home_page,
        cartographie,
        visualisation,
        ajout,
        clustering,
        graphe_comp,
        about,
    ],
    position="top",
)

# Run the selected page
pg.run()


def close_connection():
    try:
        con = st.session_state.get("duckdb_connection")
        if con:
            con.close()
    except Exception:
        pass


atexit.register(close_connection)
