import duckdb
import pandas as pd
from extract import load_jobs
from embeddings import compute_embeddings
from clustering import run_clustering
from streamlit.config import MOTHERDUCK_DATABASE

def main():
    df = load_jobs()

    embeddings = compute_embeddings(df["ml_text"].tolist())
    umap_coords, clusters = run_clustering(embeddings)
    
    result = pd.DataFrame({
        "job_id": df["job_id"],
        "cluster_id": clusters,
        "umap_x": umap_coords[:, 0],
        "umap_y": umap_coords[:, 1],
        "umap_z": umap_coords[:, 2],
    })

    con = duckdb.connect(md:job_market_RUCHE_final)

    con.execute("DROP TABLE IF EXISTS job_clusters")
    con.execute("""
        CREATE TABLE job_clusters AS
        SELECT * FROM result
    """)
    
    con.close()
    print("✅ Pipeline embeddings + clustering terminé")

if __name__ == "__main__":
    main()
