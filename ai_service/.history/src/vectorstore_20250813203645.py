import chromadb
from chromadb.utils import embedding_functions

client = chromadb.Client()
collection = client.create_collection(name="recipes")

def build_vectorstore(df, embeddings):
    for idx, emb in enumerate(embeddings):
        collection.add(
            documents=[df['chunk'].iloc[idx]],
            metadatas=[{"recipe_name": df['recipe_name'].iloc[idx]}],
            ids=[str(idx)],
            embeddings=[emb]
        )

def retrieve(query_embedding, k=5, dietary_filter=None, exclude_allergens=None):
    # Retrieve top 20 first
    results = collection.query(query_embeddings=[query_embedding], n_results=20)
    filtered_docs = []

    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        include = True
        # Filter by dietary labels
        if dietary_filter:
            if dietary_filter.lower() not in metadata.get('dietary_labels','').lower():
                include = False
        # Exclude allergens
        if exclude_allergens:
            allergens_list = metadata.get('allergens','').lower()
            for allergen in exclude_allergens:
                if allergen.lower() in allergens_list:
                    include = False
                    break
        if include:
            filtered_docs.append(doc)
        if len(filtered_docs) >= k:
            break
    return filtered_docs

