# src/embeddings.py
import os
import google.generativeai as genai

# Load GEMINI API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in your environment")

genai.configure(api_key=GEMINI_API_KEY)

def get_embedding(text: str):
    """
    Returns an embedding vector for the input text using Gemini Embeddings API.
    """
    model = genai.EmbeddingModel("textembedding-gecko-001")  # latest embedding model
    response = model.embed_text(text)
    return response.embedding
