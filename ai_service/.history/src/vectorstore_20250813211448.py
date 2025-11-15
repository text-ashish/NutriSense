# vectorstore.py
import chromadb
from chromadb.utils import embedding_functions

# Initialize Chroma client
client = chromadb.Client()
collection = client.create_collection(name="recipes")

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
            "prep_time": df.get('prep_time', '')[idx],
            "cook_time": df.get('cook_time', '')[idx],
            "total_time": df.get('total_time', '')[idx],
            "servings": df.get('servings', '')[idx],
            "nutrition_normalized": df.get('nutrition_normalized', {})[idx],
            "dietary_labels": df.get('dietary_labels', '')[idx],
            "allergens": df.get('allergens', '')[idx],
            "substitutions": df.get('substitutions', {})[idx]
        }

        collection.add(
            documents=[df['chunk'].iloc[idx]],
            metadatas=[metadata],
            ids=[str(idx)],
            embeddings=[emb]
        )

def retrieve(query_embedding, k=5, dietary_filter=None, exclude_allergens=None):
    """
    Retrieve top-k recipes from vector store based on embedding similarity.
    Apply dietary and allergen filters.
    Returns full metadata for Streamlit display.
    """
    results = collection.query(query_embeddings=[query_embedding], n_results=20)
    filtered_docs = []

    for doc_text, metadata in zip(results['documents'][0], results['metadatas'][0]):
        include = True

        # Dietary filter
        if dietary_filter:
            if dietary_filter.lower() not in metadata.get('dietary_labels','').lower():
                include = False

        # Allergen filter
        if exclude_allergens:
            allergens_list = metadata.get('allergens','').lower()
            for allergen in exclude_allergens:
                if allergen.lower() in allergens_list:
                    include = False
                    break

        if include:
            filtered_docs.append(metadata)  # return full metadata, not just text
        if len(filtered_docs) >= k:
            break

    return filtered_docs
