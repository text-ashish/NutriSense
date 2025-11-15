# app.py
import streamlit as st
import time
from src.preprocess import preprocess_recipes
from src.embeddings import get_embeddings
from src.vectorstore import build_vectorstore, retrieve, collection
from src.personalize import personalize_recipe

DATA_PATH = "data/recipes.csv"

st.set_page_config(page_title="NutriSense RAG", layout="wide")
st.title("🍽 NutriSense — Recipe & Nutrition RAG")

# -------------------------------
# Load & Cache Vectorstore
# -------------------------------
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

# -------------------------------
# User Query
# -------------------------------
st.subheader("Tell us what you’re craving")
user_description = st.text_input(
    "Example: 'low-sugar heavy breakfast', 'high-protein dinner', 'heart-friendly seafood curry'",
    placeholder="Type here..."
)

# -------------------------------
# Optional Filters
# -------------------------------
with st.expander("⚙️ Advanced Options (optional)"):
    dietary_choice = st.selectbox(
        "Dietary Preference",
        ["None", "Vegetarian", "Vegan", "Gluten-Free", "Non-Vegetarian"]
    )
    health_condition = st.selectbox(
        "Health Condition",
        ["None", "Diabetes", "Hypertension", "Heart-Friendly", "Weight Loss", 
         "PCOS", "Thyroid", "Cholesterol", "Lactose Intolerance"]
    )
    allergen_input = st.text_input("Exclude Allergens (comma-separated)")
    exclude_allergens = [a.strip() for a in allergen_input.split(",")] if allergen_input else None
    calorie_goal = st.number_input("Max Calories", min_value=0, value=0)
    protein_goal = st.number_input("Min Protein (g)", min_value=0, value=0)
    fat_goal = st.number_input("Max Fat (g)", min_value=0, value=0)

# -------------------------------
# Action Button
# -------------------------------
if st.button("🍳 Get My Recipe"):
    if not user_description.strip():
        st.warning("Please describe your meal preference.")
    else:
        # Build enhanced query
        filters_str = []
        if dietary_choice != "None":
            filters_str.append(dietary_choice)
        if health_condition != "None":
            filters_str.append(health_condition)
        if calorie_goal > 0:
            filters_str.append(f"max {calorie_goal} calories")
        if protein_goal > 0:
            filters_str.append(f"min {protein_goal}g protein")
        if fat_goal > 0:
            filters_str.append(f"max {fat_goal}g fat")

        final_query = user_description
        if filters_str:
            final_query += " | " + ", ".join(filters_str)

        start = time.time()

        # Embed & retrieve
        q_emb = get_embeddings([final_query])[0]
        results = retrieve(
    query_embedding=q_emb,
    k=6,
    dietary_filter=dietary_choice if dietary_choice != "None" else None,
    exclude_allergens=exclude_allergens,
    health_condition=health_condition if health_condition != "None" else None,
    calorie_goal=calorie_goal if calorie_goal > 0 else None,
    protein_goal=protein_goal if protein_goal > 0 else None,
    fat_goal=fat_goal if fat_goal > 0 else None
)

        results = retrieve(
            query_embedding=q_emb,
            k=6,
            dietary_filter=dietary_choice if dietary_choice != "None" else None,
            exclude_allergens=exclude_allergens,
            health_condition=health_condition if health_condition != "None" else None,
            calorie_goal=calorie_goal if calorie_goal > 0 else None,
            protein_goal=protein_goal if protein_goal > 0 else None,
            fat_goal=fat_goal if fat_goal > 0 else None
        )

        if not results:
            st.error("No matching recipes found. Try different preferences.")
        else:
            # Personalize top recipe
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
            st.caption(f"⏱ Response Time: {end-start:.2f}s")

            # ⭐ Main Recipe Display
            st.markdown("## 🍲 Your Personalized Recipe")
            st.markdown(answer)

            # 🍴 More Recipes (Compact Card View)
            if len(results) > 1:
                st.markdown("### More Recipes You Might Like")
                for r in results[1:]:
                    nut = r.get('nutrition_normalized', {})
                    calories = nut.get("calories", "N/A")
                    protein = nut.get("protein", "N/A")
                    fat = nut.get("fat", "N/A")
                    sugar = nut.get("sugar", "N/A")
                    st.markdown(
                        f"**{r.get('recipe_name', 'Unnamed')}**  "
                        f"🕒 {r.get('total_time', 'N/A')} min | "
                        f"🔥 {calories} kcal | 🥩 {protein} | 🧈 {fat} | 🍬 {sugar}"
                    )
