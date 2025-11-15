import requests

GEMINI_API_URL = "https://api.gemini.ai/v1/complete"  # hypothetical
API_KEY = "YOUR_GEMINI_PRO_KEY"

def generate_response(query, context_texts):
    prompt = f"Given the following recipes:\n{context_texts}\nAnswer the user query: {query}"
    
    response = requests.post(GEMINI_API_URL, json={
        "prompt": prompt,
        "max_output_tokens": 300
    }, headers={"Authorization": f"Bearer {API_KEY}"})
    
    return response.json().get("text", "No response")
