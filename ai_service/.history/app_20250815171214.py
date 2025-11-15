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

# Custom CSS for better styling - Eye-friendly version
custom_css = """
/* Main container styling */
.gradio-container {
    max-width: 1000px !important;
    margin: 0 auto !important;
    background: #fafbfc !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    line-height: 1.6 !important;
}

/* Header styling */
.main-header {
    background: #ffffff !important;
    color: #2d3748 !important;
    padding: 2rem !important;
    border-radius: 12px !important;
    margin-bottom: 2rem !important;
    text-align: center !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}

.main-header h1 {
    margin: 0 !important;
    font-size: 2.2rem !important;
    font-weight: 600 !important;
    color: #1a202c !important;
    letter-spacing: -0.025em !important;
}

.main-header p {
    margin: 0.75rem 0 0 0 !important;
    font-size: 1.1rem !important;
    color: #4a5568 !important;
    font-weight: 400 !important;
}

/* Card styling for sections */
.search-section, .preferences-section, .nutrition-section, .results-section {
    background: #ffffff !important;
    border-radius: 12px !important;
    padding: 2rem !important;
    margin-bottom: 1.5rem !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}

/* Section headers */
.section-header {
    color: #2d3748 !important;
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    margin-bottom: 1.5rem !important;
    padding-bottom: 0.75rem !important;
    border-bottom: 1px solid #e2e8f0 !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
}

/* Input styling */
.gradio-textbox, .gradio-dropdown, .gradio-number {
    border-radius: 8px !important;
    border: 1px solid #d1d5db !important;
    transition: all 0.2s ease !important;
    font-size: 14px !important;
    background: #ffffff !important;
}

.gradio-textbox:focus, .gradio-dropdown:focus, .gradio-number:focus {
    border-color: #3182ce !important;
    box-shadow: 0 0 0 3px rgba(49, 130, 206, 0.1) !important;
    outline: none !important;
}

/* Label styling */
label {
    color: #374151 !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    margin-bottom: 0.5rem !important;
}

/* Button styling */
.search-button {
    background: #3182ce !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.875rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: white !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
}

.search-button:hover {
    background: #2c5aa0 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15) !important;
}

/* Results styling */
.latency-display {
    background: #f7fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
    padding: 0.5rem 0.75rem !important;
    font-family: 'SF Mono', Consolas, monospace !important;
    font-size: 13px !important;
    color: #4a5568 !important;
}

.response-container {
    background: #f9fafb !important;
    border-radius: 8px !important;
    padding: 1.5rem !important;
    border: 1px solid #e5e7eb !important;
    min-height: 150px !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
}

/* Typography improvements */
h1, h2, h3 {
    color: #1a202c !important;
    font-weight: 600 !important;
}

p {
    color: #4a5568 !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}

/* Info text styling */
.info {
    color: #6b7280 !important;
    font-size: 13px !important;
}

/* Responsive design */
@media (max-width: 768px) {
    .gradio-container {
        margin: 0 1rem !important;
    }
    
    .main-header h1 {
        font-size: 1.8rem !important;
    }
    
    .search-section, .preferences-section, .nutrition-section, .results-section {
        padding: 1.25rem !important;
    }
}

/* Icons for sections */
.icon {
    font-size: 1.1rem !important;
    opacity: 0.8 !important;
}

/* Remove harsh shadows and gradients */
* {
    box-shadow: none !important;
}

.search-section, .preferences-section, .nutrition-section, .results-section {
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}

.search-button {
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
}

.search-button:hover {
    box-shadow: 0 2px 6px rgba(0,0,0,0.15) !important;
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
        
        # Hidden recipes output for function compatibility
        recipes_output = gr.Dataframe(
            visible=False,
            headers=[
                "Recipe Name", "Ingredients", "Directions", 
                "Nutrition", "Dietary labels", "Allergens", 
                "Substitutions", "Health tags"
            ]
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