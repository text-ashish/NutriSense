# vectorstore.py
import chromadb
from chromadb.utils import embedding_functions

# Initialize Chroma client
client = chromadb.Client()

# Create or get the recipes collection
collection = client.get_or_create_collection(name="recipes")

def build_vectorstore(df, embeddings):
    """
    Build vector store from DataFrame and precomputed embeddings.
    Stores full recipe metadata for richer retrieval.
    """
    for idx, emb in enumerate(embeddings):
        metadata = {
            "recipe_name": df['recipe_name'].iloc[idx],
            "ingredients": df.get('ingredients', [''])[idx],
            "directions": df.get('directions', [''])[idx],
            "prep_time": df.get('prep_time', [''])[idx],
            "cook_time": df.get('cook_time', [''])[idx],
            "total_time": df.get('total_time', [''])[idx],
            "servings": df.get('servings', [''])[idx],
            "nutrition_normalized": df.get('nutrition_normalized', [{}])[idx],
            "dietary_labels": df.get('dietary_labels', [''])[idx],
            "allergens": df.get('allergens', [''])[idx],
            "substitutions": df.get('substitutions', [{}])[idx],
            "health_tags": df.get('health_tags', [''])[idx]
        }

        collection.add(
            documents=[df['chunk'].iloc[idx]],
            metadatas=[metadata],
            ids=[str(idx)],
            embeddings=[emb]
        )

def retrieve(query_embedding, query_text=None, k=5, dietary_filter=None, exclude_allergens=None, health_condition=None):
    """
    Retrieve top-k recipes based on embedding similarity with optional filters.
    Tries exact metadata match first, then embedding search.
    """
    recipes = []

    # Step 1: Try exact name match in metadata (case-insensitive)
    if query_text:
        all_metadata = collection.get()["metadatas"]
        exact_matches = [
            m for m in all_metadata
            if m and query_text.lower() in m.get("recipe_name", "").lower()
        ]
        if exact_matches:
            recipes.extend(exact_matches)

    # Step 2: If no exact matches, fall back to embedding similarity
    if not recipes:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        recipes = results["metadatas"][0]

    # Step 3: Apply filters
    filtered_recipes = []
    for recipe in recipes:
        if not recipe:
            continue

        # Dietary filter
        if dietary_filter and dietary_filter != "None":
            if dietary_filter.lower() not in recipe.get('dietary_labels', '').lower():
                continue

        # Allergen filter
        if exclude_allergens:
            if any(a.lower() in recipe.get('ingredients', '').lower() for a in exclude_allergens):
                continue

        # Health condition filter
        if health_condition and health_condition != "None":
            if health_condition.lower() not in recipe.get('health_tags', '').lower():
                continue

        filtered_recipes.append(recipe)

    return filtered_recipes
