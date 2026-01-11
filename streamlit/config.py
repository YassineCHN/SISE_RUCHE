MOTHERDUCK_DATABASE = "job_market_RUCHE_cleaned"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DIM = 768

UMAP_PARAMS = {
    "n_components": 3,
    "n_neighbors": 15,
    "min_dist": 0.1,
    "random_state": 42,
}

HDBSCAN_PARAMS = {
    "min_cluster_size": 15,
    "metric": "euclidean",
}
