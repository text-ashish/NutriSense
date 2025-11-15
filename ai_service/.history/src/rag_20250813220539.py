import os
import streamlit as st
import google.generativeai as genai

# Load GEMINI API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Please set the GEMINI_API_KEY environment variable")
    st.stop()

# Configure Gemini client
genai.configure(api_key=GEMINI_API_KEY)

# Select a supported model
model = genai.GenerativeModel("gemini-2.5-flash")

def generate_response(query, context_texts):
    """
    Generate response using Gemini LLM for a given user query and context.
    """
    prompt = f"Given the following recipes and context:\n{context_texts}\nAnswer the user query: {query}"
    response = model.generate_content(prompt)
    return response.text if hasattr(response, "text") else "No response"

# --- Streamlit UI ---
st.title("Recipe Assistant")

user_query = st.text_input("Enter your question about the recipes:")
context_texts = st.text_area("Paste your recipe/context here:")

if st.button("Get Answer") and user_query.strip():
    answer = generate_response(user_query, context_texts)
    st.subheader("Answer:")
    st.write(answer)
