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

def retrieve(query_embedding, k=6, dietary_filter=None, exclude_allergens=None,
             health_condition=None, calorie_goal=None, protein_goal=None, fat_goal=None):

    query_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    # ✅ Flatten results into list of dicts
    results = []
    for meta, doc, rid in zip(
        query_result["metadatas"][0],
        query_result["documents"][0],
        query_result["ids"][0]
    ):
        recipe_data = dict(meta) if isinstance(meta, dict) else {}
        recipe_data["chunk"] = doc
        recipe_data["id"] = rid
        results.append(recipe_data)

    # ✅ Apply dietary filter
    if dietary_filter and dietary_filter != "None":
        results = [
            r for r in results
            if r.get("dietary_labels", "").strip().lower() == dietary_filter.strip().lower()
        ]

    # ✅ Apply allergen filter
    if exclude_allergens:
        results = [
            r for r in results
            if not any(allergen.lower() in r.get("ingredients", "").lower()
                       for allergen in exclude_allergens)
        ]

    # ✅ Health condition filter
    if health_condition and health_condition != "None":
        results = [
            r for r in results
            if health_condition.lower() in r.get("health_labels", "").lower()
        ]

    # ✅ Macro filter
    def meets_macro_goals(recipe):
        nut = recipe.get("nutrition_normalized", {})
        if calorie_goal and nut.get("calories", 0) > calorie_goal:
            return False
        if protein_goal and nut.get("protein", 0) < protein_goal:
            return False
        if fat_goal and nut.get("fat", 0) > fat_goal:
            return False
        return True

    results = [r for r in results if meets_macro_goals(r)]

    return results
