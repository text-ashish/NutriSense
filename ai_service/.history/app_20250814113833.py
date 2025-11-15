# app.py
import streamlit as st
import time
import os
import json
from src.preprocess import preprocess_recipes
from src.embeddings import get_embeddings
from src.vectorstore import build_vectorstore, retrieve, collection
from src.personalize import personalize_recipe  # NEW import

DATA_PATH = "data/recipes.csv"

st.set_page_config(page_title="NutriSense RAG", layout="wide")
st.title("🍽 NutriSense — Personalized Recipe & Nutrition RAG")


# ---------- Formatting Function ----------
def format_recipe_output(recipe_meta, recipe_name, dietary_filter, calorie_goal, carb_goal):
    substitutions = []
    try:
        subs_data = json.loads(recipe_meta.get("substitutions", "[]"))
        if isinstance(subs_data, list):
            substitutions = subs_data
    except:
        pass

    nutrition = {}
    try:
        nutrition = json.loads(recipe_meta.get("nutrition_normalized", "{}"))
    except:
        pass

    ingredients = recipe_meta.get("ingredients", "").split(",") if recipe_meta.get("ingredients") else []
    directions = recipe_meta.get("directions", "").split(". ") if recipe_meta.get("directions") else []

    output = f"""🍽 Personalized Recipe for {recipe_name} ({dietary_filter})

**Adjusted Recipe:**
- {recipe_meta.get("adjustments", "No adjustments provided")}

**Ingredients:**
{''.join([f"- {ing.strip()}\n" for ing in ingredients if ing.strip()])}

**Nutrition per serving:**
Calories: {nutrition.get('calories', 'N/A')} kcal
Protein: {nutrition.get('protein_g', 'N/A')}g
Fat: {nutrition.get('fat_g', 'N/A')}g
Carbs: {nutrition.get('carbohydrates_g', 'N/A')}g
Sugar: {nutrition.get('sugar_g', 'N/A')}g

**Why this recipe?**
- {dietary_filter}
- Meets daily goal: <{calorie_goal} calories, <{carb_goal}g carbs

**Preparation Steps:**
{''.join([f"{i+1}. {step.strip()}\n" for i, step in enumerate(directions) if step.strip()])}

**Suggested Substitutions:**
{''.join([f"- {sub}\n" for sub in substitutions])}
"""
    return output


# ---------- Prepare Data ----------
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

        q_emb = get_embeddings([user_query])[0]

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
            top_recipe = results[0]
            recipe_output = format_recipe_output(
                recipe_meta=top_recipe,
                recipe_name=top_recipe.get("recipe_name", "Unknown"),
                dietary_filter=f"{health_condition if health_condition != 'None' else ''} {dietary_choice if dietary_choice != 'None' else ''}".strip(),
                calorie_goal=calorie_goal if calorie_goal > 0 else "N/A",
                carb_goal=15  # Default carb goal from your example
            )

            end = time.time()
            st.write(f"⏱ Latency: {end-start:.2f}s")

            st.markdown(recipe_output)

            # Transparency section
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
