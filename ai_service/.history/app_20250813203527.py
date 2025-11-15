import streamlit as st
import pandas as pd
from src.preprocess import preprocess_recipes
from src.embeddings import get_embeddings
from src.vectorstore import build_vectorstore, retrieve
from src.rag import generate_response

# Load and preprocess
df = preprocess_recipes("")
embeddings = get_embeddings(df['chunk'].tolist())
build_vectorstore(df, embeddings)

st.title("Recipe & Nutrition RAG System 🍲")
user_query = st.text_input("Ask for recipes, dietary info, or substitutions:")

if user_query:
    query_emb = get_embeddings([user_query])[0]
    results = retrieve(query_emb, k=5)
    context = "\n\n".join(results['documents'][0])
    
    answer = generate_response(user_query, context)
    st.subheader("Generated Response")
    st.write(answer)
