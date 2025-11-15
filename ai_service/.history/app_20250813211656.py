import streamlit as st
import pandas as pd
import time
from src.vectorstore import retrieve, build_vectorstore
from src.rag import generate_response
from embeddings import get_embeddings
from rag_model import generate_response

# --- User Input ---
st.title("Personalized Recipe RAG System")
user_query = st.text_input("Search Recipes:")

dietary_choice = st.selectbox("Dietary Preference", ["None","Vegetarian","Vegan","Gluten-Free"])
allergen_input = st.text_input("Exclude Allergens (comma-separated)")
exclude_allergens = [a.strip() for a in allergen_input.split(",")] if allergen_input else None

# --- Retrieval & Metrics ---
if user_query:
    start = time.time()
    query_emb = get_embeddings([user_query])[0]
    results = retrieve(query_emb, k=5, dietary_filter=dietary_choice, exclude_allergens=exclude_allergens)
    answer = generate_response(user_query, "\n\n".join(results))
    end = time.time()
    
    latency = end - start
    st.write(f"⏱ Latency: {latency:.2f}s")

    # --- Display Recipes ---
    for doc in results:
        with st.expander(doc['recipe_name']):
            st.write(f"**Ingredients:** {doc['ingredients']}")
            st.write(f"**Directions:** {doc['directions']}")
            
            cols = st.columns(3)
            cols[0].metric("Calories", doc['nutrition_normalized']['calories'])
            cols[1].metric("Protein (g)", doc['nutrition_normalized']['protein_g'])
            cols[2].metric("Fat (g)", doc['nutrition_normalized']['fat_g'])
            
            st.write(f"**Substitutions:** {doc.get('substitutions','None')}")
            st.markdown(f"**Dietary Labels:** {doc['dietary_labels']}")

    # --- Download CSV ---
    recommended_df = pd.DataFrame(results)
    csv = recommended_df.to_csv(index=False)
    st.download_button(
        label="Download Recommended Recipes as CSV",
        data=csv,
        file_name='recommended_recipes.csv',
        mime='text/csv'
    )
