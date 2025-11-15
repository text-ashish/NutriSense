# app.py
import streamlit as st
from src.vectorstore import retrieve, build_vectorstore
from src.embeddings import get_embedding  # assuming you have a function for embeddings
import pandas as pd

st.set_page_config(page_title="Personalized Recipe RAG System", layout="centered")

st.title("Personalized Recipe RAG System")

# --- User Inputs ---
user_query = st.text_input("Search Recipes:", "")
dietary_pref = st.selectbox("Dietary Preference", ["", "Vegetarian", "Vegan", "Gluten-Free", "Keto"])
exclude_allergens_input = st.text_input("Exclude Allergens (comma-separated)", "")
exclude_allergens = [x.strip() for x in exclude_allergens_input.split(",") if x.strip()]

# --- Load your recipes and embeddings ---
# Example: you have a DataFrame 'df' and embeddings 'embeddings'
# Uncomment these lines if you want to rebuild vectorstore every time:
# build_vectorstore(df, embeddings)

# --- Handle Search ---
if user_query:
    query_emb = get_embedding(user_query)  # convert query to embedding
    results = retrieve(query_emb, k=5, dietary_filter=dietary_pref, exclude_allergens=exclude_allergens)
    
    if results:
        st.subheader("Matching Recipes:")
        for idx, recipe in enumerate(results, 1):
            st.markdown(f"**{idx}. {recipe.get('recipe_name', 'No Title')}**")
            st.markdown(f"- Ingredients: {recipe.get('ingredients','')}")
            st.markdown(f"- Directions: {recipe.get('directions','')}")
            st.markdown(f"- Prep Time: {recipe.get('prep_time','')} | Cook Time: {recipe.get('cook_time','')} | Total Time: {recipe.get('total_time','')}")
            st.markdown(f"- Servings: {recipe.get('servings','')}")
            st.markdown(f"- Dietary Labels: {recipe.get('dietary_labels','')}")
            st.markdown(f"- Allergens: {recipe.get('allergens','')}")
            st.markdown("---")
    else:
        st.info("No recipes found matching your criteria.")
