import streamlit as st
import duckdb
import plotly.graph_objects as go
import networkx as nx

con = duckdb.connect("md:jobs_db")

edges = con.execute(
    """
    SELECT skill_1, skill_2, weight
    FROM skill_cooccurrence
"""
).df()

st.title(" Graphe des compétences")

G = nx.Graph()
for _, row in edges.iterrows():
    G.add_edge(row.skill_1, row.skill_2, weight=row.weight)

pos = nx.spring_layout(G, dim=3, seed=42)
x, y, z, text = [], [], [], []
for node in G.nodes():
    x.append(pos[node][0])
    y.append(pos[node][1])
    z.append(pos[node][2])
    text.append(node)

fig = go.Figure(
    data=[go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=dict(size=4), text=text)]
)

st.plotly_chart(fig, use_container_width=True)
