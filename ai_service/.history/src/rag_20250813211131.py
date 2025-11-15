import os
import requests
import google.generativeai as genai

# Load GEMINI API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set the GEMINI_API_KEY environment variable")

# Configure Gemini client
genai.configure(api_key=GEMINI_API_KEY)

def generate_response(query, context_texts):
    """
    Generate response using Gemini LLM for a given user query and context.
    """
    prompt = f"Given the following recipes and context:\n{context_texts}\nAnswer the user query: {query}"

    # Use Gemini GenerativeModel
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    
    return response.text if hasattr(response, 'text') else "No response"

