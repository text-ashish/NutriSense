import gradio as gr
import time
from src.preprocess import preprocess_recipes
from src.embeddings import get_embeddings
from src.vectorstore import build_vectorstore, retrieve, collection
from src.rag import generate_response

DATA_PATH = "data/recipes.csv"

# --- Prepare data and vectorstore ---
def prepare():
    df = preprocess_recipes(DATA_PATH)
    texts = df['chunk'].tolist()
    embeddings = get_embeddings(texts)

    try:
        count = collection.count()
    except Exception:
        count = 0
    if count == 0:
        build_vectorstore(df, embeddings)
    return df

df = prepare()

# --- Main function ---
def personalized_recipe(
    user_query,
    dietary_choice,
    health_condition,
    allergen_input,
    max_calories,
    min_protein,
    max_fat
):
    if not user_query:
        return "⚠️ Please enter a recipe name or description.", "", []

    exclude_allergens = [a.strip() for a in allergen_input.split(",")] if allergen_input else None

    start = time.time()
    q_emb = get_embeddings([user_query.lower()])[0]

    # Retrieve recipes
    results = retrieve(
        query_embedding=q_emb,
        k=6,
        dietary_filter=dietary_choice,
        health_condition=health_condition,
        exclude_allergens=exclude_allergens
    )

    # Generate combined response via RAG
    answer = generate_response(
        user_query,
        retrieved_metadatas=results,
        dietary_preference=dietary_choice,
        exclude_allergens=exclude_allergens,
        health_condition=health_condition,
        calorie_goal=max_calories,
        protein_goal=min_protein,
        fat_goal=max_fat
    )

    end = time.time()
    latency = f"⏱ Latency: {end-start:.2f}s"

    # Format retrieved recipes for transparency
    formatted_recipes = []
    for r in results:
        formatted_recipes.append({
            "Recipe Name": r.get('recipe_name','Unnamed').title(),
            "Ingredients": r.get('ingredients',''),
            "Directions": r.get('directions',''),
            "Nutrition": r.get('nutrition_normalized',{}),
            "Dietary labels": r.get('dietary_labels',''),
            "Allergens": r.get('allergens',''),
            "Substitutions": r.get('substitutions',''),
            "Health tags": r.get('health_tags','')
        })

    return latency, answer, formatted_recipes

# Custom CSS for better styling
custom_css = """
/* Main container styling */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}

/* Header styling */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    padding: 2rem !important;
    border-radius: 15px !important;
    margin-bottom: 2rem !important;
    text-align: center !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2) !important;
}

.main-header h1 {
    margin: 0 !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
}

.main-header p {
    margin: 0.5rem 0 0 0 !important;
    font-size: 1.2rem !important;
    opacity: 0.9 !important;
}

/* Card styling for sections */
.search-section, .preferences-section, .nutrition-section, .results-section {
    background: white !important;
    border-radius: 15px !important;
    padding: 1.5rem !important;
    margin-bottom: 1.5rem !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08) !important;
    border: 1px solid #e1e5e9 !important;
}

/* Section headers */
.section-header {
    color: #2c3e50 !important;
    font-size: 1.3rem !important;
    font-weight: 600 !important;
    margin-bottom: 1rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 2px solid #3498db !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
}

/* Input styling */
.gradio-textbox, .gradio-dropdown, .gradio-number {
    border-radius: 10px !important;
    border: 2px solid #e1e5e9 !important;
    transition: all 0.3s ease !important;
}

.gradio-textbox:focus, .gradio-dropdown:focus, .gradio-number:focus {
    border-color: #3498db !important;
    box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1) !important;
}

/* Button styling */
.search-button {
    background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%) !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 1rem 2rem !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: white !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

.search-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 7px 20px rgba(0,0,0,0.3) !important;
}

/* Results styling */
.latency-display {
    background: #f8f9fa !important;
    border: 1px solid #dee2e6 !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    font-family: 'Courier New', monospace !important;
    color: #495057 !important;
}

.response-container {
    background: #f8f9fa !important;
    border-radius: 10px !important;
    padding: 1.5rem !important;
    border: 1px solid #e9ecef !important;
    min-height: 200px !important;
}

/* Nutrition goals styling */
.nutrition-grid {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)) !important;
    gap: 1rem !important;
}

/* Responsive design */
@media (max-width: 768px) {
    .gradio-container {
        margin: 0 1rem !important;
    }
    
    .main-header h1 {
        font-size: 2rem !important;
    }
    
    .search-section, .preferences-section, .nutrition-section, .results-section {
        padding: 1rem !important;
    }
}

/* Animation for loading states */
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.loading {
    animation: pulse 2s infinite !important;
}

/* Dataframe styling */
.gradio-dataframe {
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.1) !important;
}

/* Icons for sections */
.icon {
    font-size: 1.2rem !important;
}
"""

# --- Enhanced Gradio interface ---
with gr.Blocks(
    title="🍽 NutriSense — Personalized Recipe & Nutrition RAG",
    css=custom_css,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="green",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter")
    )
) as demo:
    
    # Header Section
    with gr.Column(elem_classes="main-header"):
        gr.HTML("""
            <h1>🍽 NutriSense</h1>
            <p>AI-Powered Personalized Recipe & Nutrition Assistant</p>
        """)
    
    # Search Section
    with gr.Column(elem_classes="search-section"):
        gr.HTML('<div class="section-header"><span class="icon">🔍</span> Recipe Search</div>')
        user_query = gr.Textbox(
            label="What are you craving today?",
            placeholder="Enter recipe name, ingredient, or description (e.g., 'chocolate cake', 'chicken breast', 'healthy breakfast')",
            lines=2,
            info="💡 Try: 'low-carb dinner', 'protein smoothie', 'comfort food'"
        )
    
    # Preferences Section
    with gr.Column(elem_classes="preferences-section"):
        gr.HTML('<div class="section-header"><span class="icon">🥗</span> Dietary Preferences & Health</div>')
        
        with gr.Row():
            with gr.Column(scale=1):
                dietary_choice = gr.Dropdown(
                    label="🌱 Dietary Preference",
                    choices=["None", "Vegetarian", "Vegan", "Gluten-Free"],
                    value="None",
                    info="Select your dietary lifestyle"
                )
                
                health_condition = gr.Dropdown(
                    label="🏥 Health Focus",
                    choices=[
                        "None", "Diabetes", "Hypertension", "Heart-Friendly",
                        "Weight Loss", "Thyroid (Hypothyroidism)", "PCOS/PCOD",
                        "Kidney-Friendly", "Liver Health", "Anemia",
                        "Bone Health (Calcium & Vitamin D)"
                    ],
                    value="None",
                    info="Choose your health priority"
                )
            
            with gr.Column(scale=1):
                allergen_input = gr.Textbox(
                    label="🚫 Allergens to Avoid",
                    placeholder="nuts, dairy, shellfish, eggs",
                    info="Separate multiple allergens with commas",
                    lines=2
                )
    
    # Nutrition Goals Section
    with gr.Column(elem_classes="nutrition-section"):
        gr.HTML('<div class="section-header"><span class="icon">📊</span> Nutritional Goals (Optional)</div>')
        gr.HTML('<p style="color: #6c757d; margin-bottom: 1rem;">Set your per-meal nutritional targets. Leave as 0 to ignore.</p>')
        
        with gr.Row():
            max_calories = gr.Number(
                label="🔥 Max Calories",
                value=0,
                minimum=0,
                maximum=2000,
                info="Maximum calories per meal"
            )
            min_protein = gr.Number(
                label="💪 Min Protein (g)",
                value=0,
                minimum=0,
                maximum=100,
                info="Minimum protein per meal"
            )
            max_fat = gr.Number(
                label="🥑 Max Fat (g)",
                value=0,
                minimum=0,
                maximum=100,
                info="Maximum fat per meal"
            )
    
    # Search Button
    with gr.Column():
        submit_btn = gr.Button(
            "🚀 Find My Perfect Recipes",
            elem_classes="search-button",
            size="lg"
        )
    
    # Results Section
    with gr.Column(elem_classes="results-section"):
        gr.HTML('<div class="section-header"><span class="icon">✨</span> Your Personalized Results</div>')
        
        # Performance indicator
        with gr.Column():
            latency_output = gr.Textbox(
                label="⚡ Processing Time",
                elem_classes="latency-display",
                interactive=False
            )
        
        # AI Response
        with gr.Column():
            gr.HTML('<h3 style="color: #2c3e50; margin-bottom: 1rem;">🤖 AI Nutritionist Recommendation</h3>')
            rag_output = gr.Markdown(
                label="Personalized Advice",
                elem_classes="response-container"
            )
    
    
    # Event binding
    submit_btn.click(
        fn=personalized_recipe,
        inputs=[
            user_query, dietary_choice, health_condition, 
            allergen_input, max_calories, min_protein, max_fat
        ],
        outputs=[latency_output, rag_output, recipes_output]
    )
    
    # Add Enter key support for search
    user_query.submit(
        fn=personalized_recipe,
        inputs=[
            user_query, dietary_choice, health_condition,
            allergen_input, max_calories, min_protein, max_fat
        ],
        outputs=[latency_output, rag_output, recipes_output]
    )

# Footer information
with gr.Blocks() as footer:
    gr.HTML("""
        <div style="text-align: center; padding: 2rem; color: #6c757d; border-top: 1px solid #dee2e6; margin-top: 2rem;">
            <p>🍽 <strong>NutriSense</strong> - Powered by AI for personalized nutrition guidance</p>
            <p style="font-size: 0.9rem; opacity: 0.7;">Always consult healthcare professionals for medical dietary advice</p>
        </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)