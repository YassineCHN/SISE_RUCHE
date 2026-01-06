import streamlit as st
import duckdb
import plotly.express as px

con = duckdb.connect("md:jobs_db")

df = con.execute("""
    SELECT
        jc.job_id,
        jc.cluster_id,
        jc.umap_x,
        jc.umap_y,
        jc.umap_z,
        jo.title,
        jo.hard_skills
    FROM job_clusters jc
    JOIN job_offers jo USING (job_id)
""").df()

st.title(" Clustering des offres")

clusters = sorted(df["cluster_id"].unique())
selected = st.multiselect("Clusters", clusters, clusters)

filtered = df[df["cluster_id"].isin(selected)]

fig = px.scatter_3d(
    filtered,
    x="umap_x",
    y="umap_y",
    z="umap_z",
    color="cluster_id",
    hover_data=["title", "hard_skills"],
    opacity=0.8
)

st.plotly_chart(fig, use_container_width=True)