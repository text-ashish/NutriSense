# vectorstore.py
import chromadb
import json
import pandas as pd
from chromadb.utils import embedding_functions
from src.embeddings import get_embeddings  # FIXED import

client = chromadb.PersistentClient(path=".chromadb")
collection = client.get_or_create_collection(name="recipes")

def build_vectorstore(df, embeddings):
    def safe_int(val):
        try:
            if pd.isna(val):
                return 0
            if isinstance(val, str) and val.strip().endswith("min"):
                return int(val.strip().split()[0])
            return int(val)
        except:
            return 0

    for idx, emb in enumerate(embeddings):
        metadata = {
            "recipe_name": str(df['recipe_name'].iloc[idx]) if not pd.isna(df['recipe_name'].iloc[idx]) else "",
            "ingredients": str(df.get('ingredients', [''])[idx]),
            "directions": str(df.get('directions', [''])[idx]),
            "prep_time": safe_int(df.get('prep_time', [0])[idx]),
            "cook_time": safe_int(df.get('cook_time', [0])[idx]),
            "total_time": safe_int(df.get('total_time', [0])[idx]),
            "servings": safe_int(df.get('servings', [0])[idx]),
            "nutrition_normalized": json.dumps(df.get('nutrition_normalized', [{}])[idx]) if not pd.isna(df.get('nutrition_normalized', [{}])[idx]) else "{}",
            "dietary_labels": json.dumps(df.get('dietary_labels', [''])[idx]) if not pd.isna(df.get('dietary_labels', [''])[idx]) else "[]",
            "allergens": json.dumps(df.get('allergens', [''])[idx]) if not pd.isna(df.get('allergens', [''])[idx]) else "[]",
            "substitutions": json.dumps(df.get('substitutions', [{}])[idx]) if not pd.isna(df.get('substitutions', [{}])[idx]) else "{}",
            "health_tags": json.dumps(df.get('health_tags', [''])[idx]) if not pd.isna(df.get('health_tags', [''])[idx]) else "[]"
        }

        collection.add(
            ids=[str(idx)],
            documents=[str(df['chunk'].iloc[idx])],
            metadatas=[metadata],
            embeddings=[emb]
        )

def retrieve(query_embedding=None, query_text=None, k=5, dietary_filter=None, exclude_allergens=None, health_condition=None):
    """
    Retrieve top-k recipes based on similarity.
    Accepts either query_embedding (list[float]) or query_text (str).
    """
    if query_embedding is None:
        if query_text is None:
            raise ValueError("You must provide either query_embedding or query_text")
        query_embedding = get_embeddings([query_text])[0]  # get first embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    filtered_recipes = []
    for i in range(len(results['ids'][0])):
        recipe_meta = results['metadatas'][0][i]

        if dietary_filter and dietary_filter.lower() != "none":
            if dietary_filter.lower() not in str(recipe_meta.get('dietary_labels', '')).lower():
                continue

        if exclude_allergens:
            allergens_list = [a.strip().lower() for a in exclude_allergens]
            ingredients_text = str(recipe_meta.get('ingredients', '')).lower()
            if any(a in ingredients_text for a in allergens_list):
                continue

        if health_condition and health_condition.lower() != "none":
            if health_condition.lower() not in str(recipe_meta.get('health_tags', '')).lower():
                continue

        filtered_recipes.append(recipe_meta)

    return filtered_recipes
