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

# Persistent client
client = chromadb.PersistentClient(path=".chromadb")

# Create or get collection
collection = client.get_or_create_collection(name="recipes")

def build_vectorstore(df, embeddings):
    """
    Build vector store from DataFrame and precomputed embeddings.
    All metadata values are converted to strings (Chroma-safe).
    """
    for idx, emb in enumerate(embeddings):
        metadata = {
            "recipe_name": str(df['recipe_name'].iloc[idx]) if not pd.isna(df['recipe_name'].iloc[idx]) else "",
            "ingredients": str(df['ingredients'].iloc[idx]) if 'ingredients' in df and not pd.isna(df['ingredients'].iloc[idx]) else "",
            "directions": str(df['directions'].iloc[idx]) if 'directions' in df and not pd.isna(df['directions'].iloc[idx]) else "",
            "prep_time": parse_time(df['prep_time'].iloc[idx]) if 'prep_time' in df else "",
            "cook_time": parse_time(df['cook_time'].iloc[idx]) if 'cook_time' in df else "",
            "total_time": parse_time(df['total_time'].iloc[idx]) if 'total_time' in df else "",
            "servings": str(df['servings'].iloc[idx]) if 'servings' in df and not pd.isna(df['servings'].iloc[idx]) else "",
            "nutrition_normalized": json.dumps(df['nutrition_normalized'].iloc[idx]) if 'nutrition_normalized' in df and not pd.isna(df['nutrition_normalized'].iloc[idx]) else "{}",
            "dietary_labels": json.dumps(df['dietary_labels'].iloc[idx]) if 'dietary_labels' in df and not pd.isna(df['dietary_labels'].iloc[idx]) else "[]",
            "allergens": json.dumps(df['allergens'].iloc[idx]) if 'allergens' in df and not pd.isna(df['allergens'].iloc[idx]) else "[]",
            "substitutions": json.dumps(df['substitutions'].iloc[idx]) if 'substitutions' in df and not pd.isna(df['substitutions'].iloc[idx]) else "{}",
            "health_tags": json.dumps(df['health_tags'].iloc[idx]) if 'health_tags' in df and not pd.isna(df['health_tags'].iloc[idx]) else "[]"
        }

        collection.add(
            ids=[str(idx)],
            documents=[str(df['chunk'].iloc[idx])],
            metadatas=[metadata],
            embeddings=[emb]
        )

def retrieve(query_embedding, k=5, dietary_filter=None, exclude_allergens=None, health_condition=None,
             calorie_goal=None, protein_goal=None, fat_goal=None):
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

        # Dietary filter
        if dietary_filter and dietary_filter.lower() != "none":
            if dietary_filter.lower() not in str(recipe_meta.get('dietary_labels', '')).lower():
                continue

        # Allergen filter
        if exclude_allergens:
            allergens_list = [a.strip().lower() for a in exclude_allergens]
            ingredients_text = str(recipe_meta.get('ingredients', '')).lower()
            if any(a in ingredients_text for a in allergens_list):
                continue

        # Health condition filter
        if health_condition and health_condition.lower() != "none":
            if health_condition.lower() not in str(recipe_meta.get('health_tags', '')).lower():
                continue

        # Nutrition goals filter
        try:
            nutrition = json.loads(recipe_meta.get('nutrition_normalized', '{}'))
        except:
            nutrition = {}
        if calorie_goal and nutrition.get('calories', 0) > calorie_goal:
            continue
        if protein_goal and nutrition.get('protein', 0) < protein_goal:
            continue
        if fat_goal and nutrition.get('fat', 0) > fat_goal:
            continue

        filtered_recipes.append(recipe_meta)

    return filtered_recipes
