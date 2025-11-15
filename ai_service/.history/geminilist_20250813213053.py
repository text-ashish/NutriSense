import google.generativeai as genai

# Configure Gemini client with your API key
genai.configure(api_key="Ge")

# List available models
models = genai.list_models()
for model in models:
    print(model.name)
