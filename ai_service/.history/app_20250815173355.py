import streamlit as st
import time
from src.preprocess import preprocess_recipes
from src.embeddings import get_embeddings
from src.vectorstore import build_vectorstore, retrieve, collection
from src.rag import generate_response

DATA_PATH = "data/recipes.csv"

st.set_page_config(page_title="NutriSense RAG", layout="wide")
st.title("🍽 NutriSense — Personalized Recipe & Nutrition RAG")

# Load data and build vectorstore once (on first run)
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
col1, col2 = st.columns([3,1])
with col1:
    user_query = st.text_input("Search Recipes (name, ingredient, or description):", value="")
with col2:
    dietary_choice = st.selectbox("Dietary Preference", ["None", "Vegetarian", "Vegan", "Gluten-Free"])
    health_condition = st.selectbox(
        "Health Condition",
        [
            "None",
            "Diabetes",
            "Hypertension",
            "Heart-Friendly",
            "Weight Loss",
            "Thyroid (Hypothyroidism)",
            "PCOS/PCOD",
            "Kidney-Friendly",
            "Liver Health",
            "Anemia",
            "Bone Health (Calcium & Vitamin D)"
        ]
    )
    allergen_input = st.text_input("Exclude Allergens (comma-separated)")
    exclude_allergens = [a.strip() for a in allergen_input.split(",")] if allergen_input else None

st.markdown("### Nutritional Goals (optional — per meal)")
col3, col4, col5 = st.columns(3)
with col3:
    calorie_goal = st.number_input("Max Calories per meal", min_value=0, value=0)
with col4:
    protein_goal = st.number_input("Min Protein (g)", min_value=0, value=0)
with col5:
    fat_goal = st.number_input("Max Fat (g)", min_value=0, value=0)

if st.button("Get Personalized Recommendation"):
    if not user_query:
        st.warning("Please enter a recipe name or description (e.g., 'jalebi' or 'low-sugar dessert').")
    else:
        start = time.time()
        # Embed query (lowercased for consistency)
        q_emb = get_embeddings([user_query.lower()])[0]

        # Retrieve recipes
        results = retrieve(
            query_embedding=q_emb,
            k=6,
            dietary_filter=dietary_choice,
            exclude_allergens=exclude_allergens,
            health_condition=health_condition
        )

        # Generate combined response via RAG
        answer = generate_response(
            user_query,
            retrieved_metadatas=results,
            dietary_preference=dietary_choice,
            exclude_allergens=exclude_allergens,
            health_condition=health_condition,
            calorie_goal=calorie_goal,
            protein_goal=protein_goal,
            fat_goal=fat_goal
        )
        end = time.time()
        st.write(f"⏱ Latency: {end-start:.2f}s")

        st.markdown("## 🍽 Personalized Response")
        st.markdown(answer)

        st.markdown("## 📖 Retrieved recipes (for transparency)")
        for r in results:
            with st.expander(r.get('recipe_name','Unnamed').title()):
                st.write("**Ingredients:**", r.get('ingredients',''))
                st.write("**Directions:**", r.get('directions',''))
                nut = r.get('nutrition_normalized',{})
                st.write("**Nutrition (per serving):**", nut)
                st.write("**Dietary labels:**", r.get('dietary_labels',''))
                st.write("**Allergens:**", r.get('allergens',''))
                st.write("**Substitutions:**", r.get('substitutions',''))
                st.write("**Health tags:**", r.get('health_tags',''))
