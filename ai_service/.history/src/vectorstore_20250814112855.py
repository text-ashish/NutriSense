# src/vectorstore.py
import chromadb
import json
import pandas as pd
from chromadb.utils import embedding_functions
from src.embeddings import get_embeddings

client = chromadb.PersistentClient(path=".chromadb")
collection = client.get_or_create_collection(name="recipes")

def build_vectorstore(df, embeddings):
    for idx, emb in enumerate(embeddings):
        metadata = {
            "recipe_name": str(df['recipe_name'].iloc[idx] or ""),
            "ingredients": str(df.get('ingredients', [''])[idx] or ""),
            "directions": str(df.get('directions', [''])[idx] or ""),
            "prep_time": int(df.get('prep_time', [0])[idx] or 0),
            "cook_time": int(df.get('cook_time', [0])[idx] or 0),
            "total_time": int(df.get('total_time', [0])[idx] or 0),
            "servings": int(df.get('servings', [0])[idx] or 0),
            "nutrition_normalized": json.dumps(df.get('nutrition_normalized', [{}])[idx] or {}),
            "dietary_labels": json.dumps(df.get('dietary_labels', [''])[idx] or ""),
            "allergens": json.dumps(df.get('allergens', [''])[idx] or ""),
            "substitutions": json.dumps(df.get('substitutions', [{}])[idx] or {}),
            "health_tags": json.dumps(df.get('health_tags', [''])[idx] or "")
        }

        collection.add(
            ids=[str(idx)],
            documents=[str(df['chunk'].iloc[idx] or "")],
            metadatas=[metadata],
            embeddings=[emb]
        )

def retrieve(
    query_text=None, query_embedding=None, k=5,
    dietary_filter=None, exclude_allergens=None, health_condition=None,
    calorie_goal=None, protein_goal=None, carb_goal=None, fat_goal=None
):
    if query_embedding is None:
        if not query_text:
            raise ValueError("Either query_text or query_embedding must be provided")
        query_embedding = get_embeddings([query_text])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    filtered = []
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]

        # Dietary filter
        if dietary_filter and dietary_filter.lower() != "none":
            if dietary_filter.lower() not in str(meta.get('dietary_labels', '')).lower():
                continue

        # Allergen filter
        if exclude_allergens:
            allergens_list = [a.strip().lower() for a in exclude_allergens]
            if any(a in str(meta.get('ingredients', '')).lower() for a in allergens_list):
                continue

        # Health filter
        if health_condition and health_condition.lower() != "none":
            if health_condition.lower() not in str(meta.get('health_tags', '')).lower():
                continue

        # Nutrition goals
        try:
            nutrition = json.loads(meta.get('nutrition_normalized', '{}'))
        except:
            nutrition = {}

        if calorie_goal and nutrition.get('calories', 0) > calorie_goal:
            continue
        if protein_goal and nutrition.get('protein_g', 0) < protein_goal:
            continue
        if carb_goal and nutrition.get('carbohydrates_g', 0) > carb_goal:
            continue
        if fat_goal and nutrition.get('fat_g', 0) > fat_goal:
            continue

        filtered.append(meta)

    return filtered
