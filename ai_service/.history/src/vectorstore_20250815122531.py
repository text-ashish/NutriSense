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

def retrieve(
    query_embedding,
    dietary_filter=None,
    health_condition=None,  # NEW
    calorie_goal=None,
    protein_goal=None,
    fat_goal=None,
    exclude_allergens=None,
    k=5
):
    query_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["metadatas", "documents", "ids"]
    )

    results = []
    for meta, doc in zip(query_result["metadatas"][0], query_result["documents"][0]):
        recipe = meta
        recipe["document"] = doc  # Keep original text

        # ✅ Dietary filter
        if dietary_filter:
            try:
                dietary_labels = json.loads(recipe.get("dietary_labels", "[]"))
                dietary_labels_str = " ".join(map(str, dietary_labels)).lower()
            except:
                dietary_labels_str = str(recipe.get("dietary_labels", "")).lower()

            if dietary_filter.lower() == "non veg" and "vegetarian" in dietary_labels_str:
                continue
            elif dietary_filter.lower() == "vegetarian" and "non vegetarian" in dietary_labels_str:
                continue

        # ✅ Health condition filter
        if health_condition:
            try:
                health_tags = json.loads(recipe.get("health_tags", "[]"))
                health_tags_str = " ".join(map(str, health_tags)).lower()
            except:
                health_tags_str = str(recipe.get("health_tags", "")).lower()

            if health_condition.lower() not in health_tags_str:
                continue

        # ✅ Allergen filter
        if exclude_allergens:
            allergens_lower = [a.strip().lower() for a in exclude_allergens]
            try:
                recipe_allergens = json.loads(recipe.get("allergens", "[]"))
                recipe_allergens_str = " ".join(map(str, recipe_allergens)).lower()
            except:
                recipe_allergens_str = str(recipe.get("allergens", "")).lower()

            if any(a in recipe_allergens_str for a in allergens_lower):
                continue

        # ✅ Macro goals
        nut = recipe.get("nutrition_normalized", {})
        if isinstance(nut, str):
            try:
                nut = json.loads(nut)
            except:
                nut = {}

        if calorie_goal and nut.get("calories") and nut["calories"] > calorie_goal:
            continue
        if protein_goal and nut.get("protein") and nut["protein"] < protein_goal:
            continue
        if fat_goal and nut.get("fat") and nut["fat"] > fat_goal:
            continue

        results.append(recipe)

    return results
