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


def normalize_label(label):
    """Normalize labels for matching: lowercase, remove spaces and hyphens."""
    return label.lower().replace("-", "").replace(" ", "")


def retrieve(
    query_embedding,
    dietary_filter=None,
    calorie_goal=None,
    protein_goal=None,
    fat_goal=None,
    exclude_allergens=None,
    health_condition=None,
    k=5
):
    query_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["metadatas", "documents"]
    )

    results = []
    for idx, (meta, doc) in enumerate(zip(query_result["metadatas"][0], query_result["documents"][0])):
        recipe_id = query_result["ids"][0][idx]
        recipe = meta
        recipe["document"] = doc
        recipe["id"] = recipe_id

        # ✅ Dietary filter
        if dietary_filter and recipe.get("dietary_labels"):
            try:
                labels = json.loads(recipe["dietary_labels"]) if isinstance(recipe["dietary_labels"], str) else recipe["dietary_labels"]
                labels_norm = [normalize_label(l) for l in labels]
                dietary_norm = normalize_label(dietary_filter)

                # Skip recipes that conflict with requested dietary choice
                if dietary_norm in ["vegetarian", "vegan"] and any(l in ["nonvegetarian", "nonveg"] for l in labels_norm):
                    continue
                if dietary_norm in ["nonvegetarian", "nonveg"] and any(l in ["vegetarian", "vegan"] for l in labels_norm):
                    continue
                if dietary_norm == "glutenfree" and "gluten" in labels_norm:
                    continue
            except:
                pass

        # ✅ Allergen filter
        if exclude_allergens:
            try:
                allergens_list = json.loads(recipe["allergens"]) if isinstance(recipe["allergens"], str) else recipe["allergens"]
                allergens_lower = [a.strip().lower() for a in allergens_list]
                if any(a.lower() in allergens_lower for a in exclude_allergens):
                    continue
            except:
                pass

        # ✅ Health condition filter
        if health_condition:
            try:
                tags = json.loads(recipe["health_tags"]) if isinstance(recipe["health_tags"], str) else recipe["health_tags"]
                tags_lower = [t.lower() for t in tags]
                if normalize_label(health_condition) not in [normalize_label(t) for t in tags_lower]:
                    continue
            except:
                pass

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
