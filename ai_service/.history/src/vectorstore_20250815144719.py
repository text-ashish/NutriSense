import chromadb
import json
import pandas as pd
from chromadb.utils import embedding_functions

# --- Helper: convert time strings to minutes ---
def parse_time(value):
    """Convert '10 min', '1 hr 20 min', or numeric values to int minutes. Returns string for Chroma."""
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float)):
        return str(int(value))
    if isinstance(value, str):
        value = value.lower().strip()
        total_minutes = 0
        # Extract hours
        if "hr" in value:
            try:
                hours = int(''.join(ch for ch in value.split("hr")[0] if ch.isdigit()))
                total_minutes += hours * 60
                value = value.split("hr")[1]
            except ValueError:
                pass
        # Extract minutes
        if "min" in value:
            try:
                minutes = int(''.join(ch for ch in value.split("min")[0] if ch.isdigit()))
                total_minutes += minutes
            except ValueError:
                pass
        return str(total_minutes) if total_minutes > 0 else ""
    return ""

# Persistent client using DuckDB+Parquet (works on Streamlit Cloud)
client = Client(
    persist_directory="db",          # optional
    chroma_db_impl="duckdb+parquet"  # DuckDB backend
)

# Create or get collection
collection = client.get_or_create_collection(name="recipes")

def build_vectorstore(df, embeddings):
    """
    Build vector store from DataFrame and precomputed embeddings.
    All metadata values are converted to lowercase strings (Chroma-safe).
    """
    # Normalize text fields
    df['recipe_name'] = df['recipe_name'].str.lower()
    df['ingredients'] = df.get('ingredients', pd.Series(['']*len(df))).str.lower()
    df['directions'] = df.get('directions', pd.Series(['']*len(df))).str.lower()

    for idx, emb in enumerate(embeddings):
        metadata = {
            "recipe_name": str(df['recipe_name'].iloc[idx]) if not pd.isna(df['recipe_name'].iloc[idx]) else "",
            "ingredients": str(df.get('ingredients', [''])[idx]),
            "directions": str(df.get('directions', [''])[idx]),
            "prep_time": parse_time(df.get('prep_time', [''])[idx]),
            "cook_time": parse_time(df.get('cook_time', [''])[idx]),
            "total_time": parse_time(df.get('total_time', [''])[idx]),
            "servings": str(df.get('servings', [''])[idx]) if not pd.isna(df.get('servings', [''])[idx]) else "",
            "nutrition_normalized": json.dumps(df.get('nutrition_normalized', [{}])[idx]) if not pd.isna(df.get('nutrition_normalized', [{}])[idx]) else "{}",
            "dietary_labels": json.dumps(df.get('dietary_labels', [''])[idx]).lower() if not pd.isna(df.get('dietary_labels', [''])[idx]) else "[]",
            "allergens": json.dumps(df.get('allergens', [''])[idx]).lower() if not pd.isna(df.get('allergens', [''])[idx]) else "[]",
            "substitutions": json.dumps(df.get('substitutions', [{}])[idx]) if not pd.isna(df.get('substitutions', [{}])[idx]) else "{}",
            "health_tags": json.dumps(df.get('health_tags', [''])[idx]).lower() if not pd.isna(df.get('health_tags', [''])[idx]) else "[]"
        }

        collection.add(
            ids=[str(idx)],
            documents=[str(df['chunk'].iloc[idx]).lower()],
            metadatas=[metadata],
            embeddings=[emb]
        )

def retrieve(query_embedding, k=5, dietary_filter=None, exclude_allergens=None, health_condition=None):
    """
    Retrieve top-k recipes based on embedding similarity with optional filters.
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    filtered_recipes = []
    for i in range(len(results['ids'][0])):
        recipe_meta = results['metadatas'][0][i]

        # Lowercase for consistent matching
        recipe_ingredients = str(recipe_meta.get('ingredients', '')).lower()
        recipe_dietary = str(recipe_meta.get('dietary_labels', '')).lower()
        recipe_health = str(recipe_meta.get('health_tags', '')).lower()
        recipe_name = str(recipe_meta.get('recipe_name', '')).lower()

        # Dietary filter
        if dietary_filter and dietary_filter.lower() != "none":
            if dietary_filter.lower() not in recipe_dietary:
                continue

        # Allergen filter
        if exclude_allergens:
            allergens_list = [a.strip().lower() for a in exclude_allergens]
            if any(a in recipe_ingredients for a in allergens_list):
                continue

        # Health condition filter
        if health_condition and health_condition.lower() != "none":
            if health_condition.lower() not in recipe_health:
                continue

        filtered_recipes.append(recipe_meta)

    return filtered_recipes
