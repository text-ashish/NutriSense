import os
import google.generativeai as genai

# Load GEMINI API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set the GEMINI_API_KEY environment variable")

# Configure Gemini client
genai.configure(api_key=GEMINI_API_KEY)

def generate_response(query, context_texts, max_output_tokens=300):
    """
    Generate response using Gemini LLM for a given user query and context.

    Args:
        query (str): User query or question.
        context_texts (str): Concatenated context from retrieved recipes.
        max_output_tokens (int): Maximum tokens to generate.

    Returns:
        str: Generated response text.
    """
    prompt = f"Given the following recipes and context:\n{context_texts}\nAnswer the user query: {query}"

    # Use Gemini GenerativeModel
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(
        prompt=prompt,
        max_output_tokens=max_output_tokens
    )
    
    # Gemini response may contain 'text' attribute
    if hasattr(response, 'text') and response.text:
        return response.text.strip()
    elif hasattr(response, 'content') and response.content:
        # fallback if 'content' attribute exists
        return response.content.strip()
    else:
        return "No response generated"
