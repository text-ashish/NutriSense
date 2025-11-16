# src/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

# load once globally
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(EMBED_MODEL_NAME)

def get_embeddings(texts):
    """
    texts: list[str]
    returns: list[numpy.ndarray]
    """
    embs = _model.encode(texts, show_progress_bar=False)
    return embs.tolist()
