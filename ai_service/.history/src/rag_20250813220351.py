import os
import google.generativeai as genai

# Load GEMINI API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set the GEMINI_API_KEY environment variable")

# Configure Gemini client
genai.configure(api_key=GEMINI_API_KEY)

# Use a valid Gemini model for text generation
model = genai.GenerativeModel("gemini-2.5-flash")  # pick a supported model

def generate_response(query, context_texts):
    """
    Generate response using Gemini LLM for a given user query and context.
    """
    prompt = f"Given the following recipes and context:\n{context_texts}\nAnswer the user query: {query}"

    # Directly pass the prompt string
    response = model.generate_content(prompt)

    # Return the generated text
    return response.text if hasattr(response, "text") else "No response"
