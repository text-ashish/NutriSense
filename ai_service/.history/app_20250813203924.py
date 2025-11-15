import streamlit as st
import pandas as pd
from utils.normalize import normalize_nutrition
from utils.embeddings import get_embeddings
from utils.vectorstore import retrieve
from utils.evaluation import measure_latency

# Load recipes
recipes_df = pd.read_csv("data/recipes.csv")
for i, row in recipes_df.iterrows():
    recipes_df.loc[i, 'nutrition_normalized'] = normalize_nutrition(row['nutrition'])

# User query
user_query = st.text_input("Enter your recipe query:")

# Filters
dietary_choice = st.selectbox("Dietary Preference", ["None","Vegetarian","Vegan","Gluten-Free"])
allergen_input = st.text_input("Exclude Allergens (comma-separated)")
exclude_allergens = [a.strip() for a in allergen_input.split(",")] if allergen_input else None

if user_query:
    query_emb = get_embeddings([user_query])[0]
    results = retrieve(query_emb, k=5, dietary_filter=dietary_choice, exclude_allergens=exclude_allergens)

    for doc in results:
        st.markdown(f"**{doc['recipe_name']}**")
        st.write(doc['ingredients'])
        st.write(doc['directions'])
        st.write(doc['nutrition_normalized'])
        st.write(doc.get('substitutions', "No substitutions"))
