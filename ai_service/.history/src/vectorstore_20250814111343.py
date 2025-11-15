# vectorstore.py
import chromadb

from chromadb.utils import embedding_functions

# Persistent client (stores data locally in .chromadb folder)
client = chromadb.PersistentClient(path=".chromadb")

# Create or get collection
collection = client.get_or_create_collection(name="recipes")

def build_vectorstore(df, embeddings):
    """
    Build vector store from DataFrame and precomputed embeddings.
    Convert all metadata values to Chroma-supported types.
    """
    for idx, emb in enumerate(embeddings):
        metadata = {
            "recipe_name": str(df['recipe_name'].iloc[idx]),
            "ingredients": str(df.get('ingredients', [''])[idx]),
            "directions": str(df.get('directions', [''])[idx]),
            "prep_time": int(df.get('prep_time', [0])[idx]) if not pd.isna(df.get('prep_time', [0])[idx]) else None,
            "cook_time": int(df.get('cook_time', [0])[idx]) if not pd.isna(df.get('cook_time', [0])[idx]) else None,
            "total_time": int(df.get('total_time', [0])[idx]) if not pd.isna(df.get('total_time', [0])[idx]) else None,
            "servings": int(df.get('servings', [0])[idx]) if not pd.isna(df.get('servings', [0])[idx]) else None,
            "nutrition_normalized": json.dumps(df.get('nutrition_normalized', [{}])[idx]),
            "dietary_labels": json.dumps(df.get('dietary_labels', [''])[idx]),
            "allergens": json.dumps(df.get('allergens', [''])[idx]),
            "substitutions": json.dumps(df.get('substitutions', [{}])[idx]),
            "health_tags": json.dumps(df.get('health_tags', [''])[idx])
        }

        collection.add(
            ids=[str(idx)],
            documents=[str(df['chunk'].iloc[idx])],
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

        filtered_recipes.append(recipe_meta)

    return filtered_recipes
