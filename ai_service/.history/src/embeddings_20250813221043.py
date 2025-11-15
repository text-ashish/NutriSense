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
    response = genai.embed_text(
        model="textembedding-gecko-001",
        text=text
    )
    # The embedding vector is in response["embedding"]
    return response["embedding"]
