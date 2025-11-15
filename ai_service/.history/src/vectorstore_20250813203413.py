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

def retrieve(query_embedding, k=5):
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return results
