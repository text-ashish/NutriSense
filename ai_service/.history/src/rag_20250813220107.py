import os
import google.generativeai as genai

# Load GEMINI API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set the GEMINI_API_KEY environment variable")

# Configure Gemini client
genai.configure(api_key=GEMINI_API_KEY)

# Load the most efficient text model
model = genai.GenerativeModel('gemini-2.5-flash')

def generate_response(query, context_texts, max_output_tokens=300):
    """
    Generate response using Gemini LLM for a given user query and recipe context.
    """
    prompt = f"Given the following recipes and context:\n{context_texts}\nAnswer the user query: {query}"

    # Generate content from the model
    response = model.generate_content(
        parts=[{"text": prompt}],
        max_output_tokens=max_output_tokens,
        temperature=0.7  # optional, controls creativity
    )

    # Return text output
    return response.text if hasattr(response, 'text') else "No response"
