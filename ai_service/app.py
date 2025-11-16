# app.py
from dotenv import load_dotenv
import os
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables at the top
load_dotenv() 

# --- Local Module Imports ---
from src.preprocess import preprocess_recipes
from src.embeddings import get_embeddings
from src.vectorstore import build_vectorstore, retrieve, collection
from src.rag import generate_response

# --- Initialize FastAPI App (Only Once) ---
app = FastAPI(title="NutriSense API")

# --- CORS Middleware ---
# This server is only called by your Node.js server, not the browser directly.
# Setting origins to "*" is safe and simple for this internal communication.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = "data/recipes.csv"

# --- Prepare data ---
def prepare():
    df = preprocess_recipes(DATA_PATH)
    texts = df['chunk'].tolist()
    embeddings = get_embeddings(texts)
    print("⚙️ Rebuilding vector store from scratch...")
    build_vectorstore(df, embeddings)

    return df

df = prepare()
print("✅ Data preparation complete.")


# --- Main function ---
def personalized_recipe(query, dietary, health, allergens, calories, protein, fat):
    if not query:
        return "⚠️ Please enter a recipe name or description."

    exclude_allergens = [a.strip() for a in allergens.split(",")] if allergens else None
    start_time = time.time()
    
    query_emb = get_embeddings([query.lower()])[0]

    retrieved = retrieve(
        query_embedding=query_emb,
        k=6,
        dietary_filter=dietary,
        exclude_allergens=exclude_allergens,
        health_condition=health
    )

    answer = generate_response(
        query,
        retrieved_metadatas=retrieved,
        dietary_preference=dietary,
        exclude_allergens=exclude_allergens,
        health_condition=health,
        calorie_goal=calories,
        protein_goal=protein,
        fat_goal=fat
    )

    latency = round(time.time() - start_time, 2)
    return {"latency": latency, "recommendation": answer}

# --- FastAPI setup ---
class RecipeRequest(BaseModel):
    query: str
    dietary: str = "None"
    health: str = "None"
    allergens: str = ""
    calories: float = 0
    protein: float = 0
    fat: float = 0

@app.post("/get_recipe")
def get_recipe(request: RecipeRequest):
    # --- NEW DEBUG PRINT ---
    print("[Python DEBUG] Checkpoint 4: /get_recipe endpoint hit successfully.")
    return personalized_recipe(
        request.query,
        request.dietary,
        request.health,
        request.allergens,
        request.calories,
        request.protein,
        request.fat
    )

@app.get("/")
def read_root():
    return {"status": "NutriSense API is running"}

    # uvicorn app:app --reload --host 0.0.0.0 --port 8001
