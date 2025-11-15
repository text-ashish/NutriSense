# src/vectorstore.py
import chromadb
from chromadb.config import Settings
import os

# Initialize Chroma client (use in-memory by default)
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=".chromadb"))

# get_or_create collection
collection = client.get_or_create_collection(name="recipes")

def build_vectorstore(df, embeddings):
    """
    df: DataFrame from preprocess_recipes
    embeddings: list of vectors matching df rows
    """
    # optional: clear existing collection
    # collection.delete()  # careful - deletes whole collection
    ids = [str(i) for i in range(len(embeddings))]
    docs = df['chunk'].tolist()
    metas = []
    for idx, row in df.iterrows():
        metas.append({
            "recipe_name": row['recipe_name'],
            "ingredients": row['ingredients'],
            "directions": row['directions'],
            "prep_time": row['prep_time'],
            "cook_time": row['cook_time'],
            "total_time": row['total_time'],
            "servings": row['servings'],
            "nutrition_normalized": row['nutrition_normalized'],
            "dietary_labels": row['dietary_labels'],
            "allergens": row['allergens'],
            "substitutions": row['substitutions'],
            "health_tags": row['health_tags']
        })
    # add in batch
    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

def retrieve(query_embedding, query_text=None, k=5, dietary_filter=None, exclude_allergens=None, health_condition=None, calorie_goal=None, protein_goal=None, fat_goal=None):
    """
    1) Try exact name match in metadata (case-insensitive)
    2) Else run semantic retrieval
    3) Apply filters (dietary, allergens, health condition)
    4) Apply nutritional goal enforcement (if goals provided) - will rank/flag
    Returns: list of metadata dicts
    """
    results_meta = []

    # Step 1: exact name match
    if query_text:
        all_meta = collection.get()['metadatas']  # list of lists per collection slice
        # all_meta is like [[meta0,meta1,...]] ; flatten
        if isinstance(all_meta, list):
            flat = []
            for sub in all_meta:
                if sub:
                    flat.extend(sub)
            for m in flat:
                if not m:
                    continue
                name = m.get('recipe_name','').lower()
                if query_text.lower().strip() in name:
                    results_meta.append(m)
    # Step 2: semantic search if no exact or to augment
    if not results_meta:
        res = collection.query(query_embeddings=[query_embedding], n_results=k)
        # res['metadatas'] is list with one sublist
        candidates = res.get('metadatas', [[]])[0]
        results_meta = candidates

    filtered = []
    for m in results_meta:
        if not m:
            continue

        # dietary filter
        if dietary_filter and dietary_filter != "None":
            if dietary_filter.lower() not in m.get('dietary_labels','').lower():
                continue

        # allergen exclude
        if exclude_allergens:
            ing = m.get('ingredients','').lower()
            skip = False
            for a in exclude_allergens:
                if a and a.lower() in ing:
                    skip = True
                    break
            if skip:
                continue

        # health condition filter
        if health_condition and health_condition != "None":
            if health_condition.lower() not in m.get('health_tags','').lower():
                continue

        # nutritional goal filtering (soft): you may still keep but mark
        nut = m.get('nutrition_normalized', {})
        # if goal provided and recipe far exceeds, skip (simple policy)
        if calorie_goal and calorie_goal > 0:
            if nut.get('calories',0) > calorie_goal:
                # skip very high calorie items when goal is small
                # keep if within 150% of goal though
                if nut.get('calories',0) > calorie_goal * 1.5:
                    continue
        if protein_goal and protein_goal > 0:
            # if recipe has extremely low protein compared to desired single-recipe target, skip
            if nut.get('protein_g',0) < protein_goal * 0.2:  # e.g., recipe provides <20% of protein target
                pass  # we allow but can flag later

        filtered.append(m)

    # Limit to k
    return filtered[:k]
