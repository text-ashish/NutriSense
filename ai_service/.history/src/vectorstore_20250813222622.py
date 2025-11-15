# vectorstore.py
import chromadb
from chromadb.utils import embedding_functions

# Initialize Chroma client
client = chromadb.Client()

# Use get_or_create=True to avoid "collection already exists" errors
collection = client.create_collection(name="recipes", get_or_create=True)

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
            "substitutions": df.get('substitutions', [{}])[idx]
        }

        collection.add(
            documents=[df['chunk'].iloc[idx]],
            metadatas=[metadata],
            ids=[str(idx)],
            embeddings=[emb]
        )

def retrieve(query_embedding, k=5, dietary_filter=None, exclude_allergens=None, health_condition=None):
    """
    Retrieve top-k recipes based on embedding similarity with optional filters:
    - dietary_filter: "Vegetarian", "Vegan", etc.
    - exclude_allergens: list of allergens to avoid
    - health_condition: e.g., "Diabetes", "Hypertension"
    """
    # Assume 'recipes_db' is a list of dicts with recipe info
    filtered_recipes = []

    for recipe in recipes_db:
        # Dietary filter
        if dietary_filter and dietary_filter != "None":
            if dietary_filter not in recipe['dietary_labels']:
                continue

        # Allergen filter
        if exclude_allergens:
            if any(a in recipe['ingredients'] for a in exclude_allergens):
                continue

        # Health condition filter
        if health_condition and health_condition != "None":
            if 'health_tags' not in recipe or health_condition not in recipe['health_tags']:
                continue

        filtered_recipes.append(recipe)

    # TODO: Apply vector similarity ranking here using query_embedding
    # For now, just return top-k
    return filtered_recipes[:k]