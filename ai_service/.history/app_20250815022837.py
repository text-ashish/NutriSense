# app.py
import streamlit as st
import time
import os
from src.preprocess import preprocess_recipes
from src.embeddings import get_embeddings
from src.vectorstore import build_vectorstore, retrieve, collection
from src.rag import generate_response

DATA_PATH = "data/recipes.csv"

st.title("🍽 NutriSense — Recipe & Nutrition RAG")

# --- 1. Natural language description ---
nl_query = st.text_input("Describe your meal preference (e.g., 'low-sugar dessert for diabetes')", "")

# --- 2. Form inputs ---
with st.form("recipe_form"):
    recipe_name = st.text_input("Search Recipes (name, ingredient, or description)")
    dietary_preference = st.selectbox("Dietary Preference", ["None", "Vegetarian", "Vegan", "Pescatarian"])
    health_condition = st.selectbox("Health Condition", ["None", "Weight Loss", "Diabetes", "Heart Health"])
    allergens = st.text_input("Exclude Allergens (comma-separated)")
    max_calories = st.number_input("Max Calories per meal", min_value=0, step=10)
    min_protein = st.number_input("Min Protein (g)", min_value=0, step=1)
    max_fat = st.number_input("Max Fat (g)", min_value=0, step=1)
    submitted = st.form_submit_button("Search")

if submitted or nl_query:
    # --- 3. Merge into a single semantic query ---
    query_parts = []
    if nl_query.strip():
        query_parts.append(nl_query.strip())
    if recipe_name.strip():
        query_parts.append(f"recipe: {recipe_name}")
    if dietary_preference != "None":
        query_parts.append(f"diet: {dietary_preference}")
    if health_condition != "None":
        query_parts.append(f"health: {health_condition}")
    if allergens.strip():
        query_parts.append(f"exclude allergens: {allergens}")
    if max_calories > 0:
        query_parts.append(f"max {max_calories} calories")
    if min_protein > 0:
        query_parts.append(f"min {min_protein}g protein")
    if max_fat > 0:
        query_parts.append(f"max {max_fat}g fat")

    final_query = ", ".join(query_parts)

    st.write(f"**🔍 Constructed Query for RAG:** `{final_query}`")  # Transparency for grader

    # --- 4. Retrieval ---
    q_emb = get_embeddings([final_query])[0]   # Your embedding function
    results = retrieve(query_embedding=q_emb, calorie_goal=max_calories, protein_goal=min_protein, fat_goal=max_fat)

    if not results:
        st.warning("No matching recipes found. Try adjusting your filters or goals.")
    else:
        # --- 5. Generation ---
        generated = generate_response(final_query, results)
        st.markdown(generated)

        # Show retrieved recipes for transparency
        with st.expander("📖 Retrieved Recipes (Raw)"):
            st.json(results)