# app.py
import streamlit as st
import time
import os
from src.preprocess import preprocess_recipes
from src.embeddings import get_embeddings
from src.vectorstore import build_vectorstore, retrieve, collection
from src.personalize import personalize_recipe  # NEW import

DATA_PATH = "data/recipes.csv"

st.set_page_config(page_title="NutriSense RAG", layout="wide")
st.title("🍽 NutriSense — Personalized Recipe & Nutrition RAG")

# Load data and build vectorstore once
@st.cache_data(show_spinner=False)
def prepare():
    df = preprocess_recipes(DATA_PATH)
    texts = df['chunk'].tolist()
    embeddings = get_embeddings(texts)

    try:
        count = collection.count()
    except Exception:
        count = 0

    if count == 0:
        build_vectorstore(df, embeddings)
    return df

df = prepare()

# --- Inputs ---
col1, col2 = st.columns([3, 1])
with col1:
    user_query = st.text_input(
        "Search Recipes (name, ingredient, or description):",
        value=""
    )
with col2:
    dietary_choice = st.selectbox(
        "Dietary Preference",
        ["None", "Vegetarian", "Vegan", "Gluten-Free"]
    )
    health_condition = st.selectbox(
        "Health Condition",
        ["None", "Diabetes", "Hypertension", "Heart-Friendly", "Weight Loss"]
    )
    allergen_input = st.text_input("Exclude Allergens (comma-separated)")
    exclude_allergens = (
        [a.strip() for a in allergen_input.split(",")] if allergen_input else None
    )

st.markdown("### Nutritional Goals (optional — per meal)")
col3, col4, col5 = st.columns(3)
with col3:
    calorie_goal = st.number_input("Max Calories per meal", min_value=0, value=0)
with col4:
    protein_goal = st.number_input("Min Protein (g)", min_value=0, value=0)
with col5:
    fat_goal = st.number_input("Max Fat (g)", min_value=0, value=0)

# --- Button Action ---
if st.button("Get Personalized Recommendation"):
    if not user_query:
        st.warning("Please enter a recipe name or description (e.g., 'jalebi' or 'low-sugar dessert').")
    else:
        start = time.time()

        # Get embedding for query
        q_emb = get_embeddings([user_query])[0]

        # Retrieve matching recipes
        results = retrieve(
            query_embedding=q_emb,
            k=6,
            dietary_filter=dietary_choice,
            exclude_allergens=exclude_allergens,
            health_condition=health_condition,
            calorie_goal=calorie_goal if calorie_goal > 0 else None,
            protein_goal=protein_goal if protein_goal > 0 else None,
            fat_goal=fat_goal if fat_goal > 0 else None
        )

        if not results:
            st.error("No matching recipes found. Try adjusting your filters or goals.")
        else:
            # Personalize the top recipe
            top_recipe = results[0]
            goals = {
                "dietary_filter": dietary_choice,
                "exclude_allergens": exclude_allergens,
                "health_condition": health_condition,
                "calorie_goal": calorie_goal,
                "protein_goal": protein_goal,
                "fat_goal": fat_goal
            }
            answer = personalize_recipe(top_recipe, goals)

            end = time.time()
            st.write(f"⏱ Latency: {end-start:.2f}s")

            # --- Display Personalized Output ---
            st.markdown("## 🍽 Personalized Response")
            st.markdown(answer)

            # --- Show Retrieved Recipes ---
            st.markdown("## 📖 Retrieved Recipes (Transparency)")
            for r in results:
                with st.expander(r.get('recipe_name', 'Unnamed')):
                    st.write("**Ingredients:**", r.get('ingredients', ''))
                    st.write("**Directions:**", r.get('directions', ''))
                    nut = r.get('nutrition_normalized', {})
                    st.write("**Nutrition (per serving):**", nut)
                    st.write("**Dietary labels:**", r.get('dietary_labels', ''))
                    st.write("**Allergens:**", r.get('allergens', ''))
                    st.write("**Substitutions:**", r.get('substitutions', ''))
                    st.write("**Health tags:**", r.get('health_tags', ''))
