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
    """
    prompt_text = f"Given the following recipes and context:\n{context_texts}\nAnswer the user query: {query}"

    # Directly use genai.generate_text()
    response = genai.generate_text(
        model="gemini-2.5-flash",
        prompt=prompt_text,
        max_output_tokens=max_output_tokens
    )

    # `response` is a dict; the generated text is under 'candidates'
    if response and "candidates" in response and len(response["candidates"]) > 0:
        return response["candidates"][0]["content"]
    else:
        return "No response"
