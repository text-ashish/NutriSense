import streamlit as st
import pandas as pd
import time
from src.embeddings import get_embeddings
from src.vectorstore import retrieve
from src.rag import generate_response

# --- App Title ---
st.title("🍽 Personalized Recipe RAG System")

# --- User Inputs ---
user_query = st.text_input("Search Recipes:")

dietary_choice = st.selectbox(
    "Dietary Preference", ["None", "Vegetarian", "Vegan", "Gluten-Free"]
)

allergen_input = st.text_input("Exclude Allergens (comma-separated)")
exclude_allergens = [a.strip() for a in allergen_input.split(",")] if allergen_input else None

health_condition = st.selectbox(
    "Health Condition",
    ["None", "Diabetes", "Hypertension", "Heart-Friendly", "Weight Loss"]
)

st.write("### Nutritional Goals (Optional)")
calorie_goal = st.number_input("Daily Calories", min_value=0, value=0)
protein_goal = st.number_input("Protein (g)", min_value=0, value=0)
fat_goal = st.number_input("Fat (g)", min_value=0, value=0)

# --- Retrieval & Response ---
if user_query:
    start = time.time()

    # Get query embedding
    query_emb = get_embeddings([user_query])[0]

    # Retrieve recipes with filters
    results = retrieve(
    query_embedding=query_emb,
    query_text=user_query,
    k=5,
    dietary_filter=dietary_choice,
    exclude_allergens=exclude_allergens,
    health_condition=health_condition   
)

    # Generate AI response
    answer = generate_response(user_query, "\n\n".join([r['recipe_name'] for r in results]))

    end = time.time()
    latency = end - start
    st.write(f"⏱ Latency: {latency:.2f}s")

    # --- Display Personalized Response ---
    st.markdown("### 🍽 Personalized Response")
    st.write(answer)

    # --- Display Recipes ---
    st.markdown("### 📖 Recommended Recipes")
    for doc in results:
        with st.expander(doc['recipe_name']):
            st.write(f"**Ingredients:** {doc['ingredients']}")
            st.write(f"**Directions:** {doc['directions']}")
            
            # Nutrition Metrics
            cols = st.columns(3)
            cols[0].metric("Calories", doc['nutrition_normalized']['calories'])
            cols[1].metric("Protein (g)", doc['nutrition_normalized']['protein_g'])
            cols[2].metric("Fat (g)", doc['nutrition_normalized']['fat_g'])

            # Substitutions & Labels
            st.write(f"**Substitutions:** {doc.get('substitutions','None')}")
            st.markdown(f"**Dietary Labels:** {doc['dietary_labels']}")
            st.markdown(f"**Health Tags:** {doc.get('health_tags', 'None')}")

            # Goal Tracking Highlight
            goal_feedback = []
            if calorie_goal and doc['nutrition_normalized']['calories'] > calorie_goal:
                goal_feedback.append("⚠ Exceeds calorie goal")
            if protein_goal and doc['nutrition_normalized']['protein_g'] < protein_goal:
                goal_feedback.append("⚠ Below protein goal")
            if fat_goal and doc['nutrition_normalized']['fat_g'] > fat_goal:
                goal_feedback.append("⚠ Exceeds fat goal")
            if goal_feedback:
                st.warning(", ".join(goal_feedback))
