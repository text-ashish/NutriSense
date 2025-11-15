import google.generativeai as genai

# Configure Gemini client with your API key
genai.configure(api_key="GEMINI_API_KEY")

# List available models
models = genai.list_models()
for model in models:
    print(model.name)
