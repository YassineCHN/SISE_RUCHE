import umap
import hdbscan
import pandas as pd
from streamlit.config import UMAP_PARAMS, HDBSCAN_PARAMS

def run_clustering(embeddings):
    reducer = umap.UMAP(**UMAP_PARAMS)
    umap_coords = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(**HDBSCAN_PARAMS)
    clusters = clusterer.fit_predict(embeddings)

    return umap_coords, clusters