from sentence_transformers import SentenceTransformer
import numpy as np
from config import EMBEDDING_MODEL

def compute_embeddings(texts):
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return np.array(embeddings)