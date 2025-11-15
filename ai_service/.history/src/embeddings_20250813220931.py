from sentence_transformers import SentenceTransformer

# Use HuggingFace model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(texts):
    return model.encode(texts, show_progress_bar=True)
