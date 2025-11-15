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
        exclude_allergens=exclude_allergens,
        health_condition=health_condition
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

# Custom CSS for enhanced styling
custom_css = """
/* Main theme and background */
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

/* Header styling */
.main-header {
    text-align: center;
    background: rgba(255, 255, 255, 0.95);
    padding: 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.main-header h1 {
    font-size: 2.5rem;
    background: linear-gradient(45deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    font-weight: 700;
}

.subtitle {
    color: #666;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

/* Card styling */
.search-card, .nutrition-card, .results-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Input field styling */
.gradio-textbox, .gradio-dropdown, .gradio-number {
    border-radius: 12px !important;
    border: 2px solid #e1e5e9 !important;
    transition: all 0.3s ease !important;
}

.gradio-textbox:focus, .gradio-dropdown:focus, .gradio-number:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    transform: translateY(-2px);
}

/* Button styling */
.submit-btn {
    background: #000 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
}

.submit-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
}

/* Results styling */
.results-section {
    margin-top: 2rem;
}



/* Responsive design */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2rem;
    }
    
    .search-card, .nutrition-card, .results-card {
        padding: 1rem;
    }
}

/* Animation for loading states */
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
}

.loading {
    animation: pulse 2s infinite;
}

/* Emoji styling */
.emoji {
    font-size: 1.2em;
    margin-right: 0.5rem;
}

/* Section headers */
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #333;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
}

/* Nutrition goals styling */
.nutrition-goals {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
}

.nutrition-goals h3 {
    color: white;
    margin-bottom: 1rem;
}
"""

# --- Enhanced Gradio interface ---
with gr.Blocks(
    title="🍽 NutriSense — Personalized Recipe & Nutrition RAG",
    css=custom_css,
    theme=gr.themes.Soft()
) as demo:
    
    # Header Section
    with gr.Column(elem_classes="main-header"):
        gr.HTML("""
            <h1>🍽️ NutriSense</h1>
            <p class="subtitle">Your AI-Powered Personal Nutrition & Recipe Assistant</p>
            <p style="color: #888; margin-top: 1rem;">Discover personalized recipes tailored to your dietary preferences, health conditions, and nutritional goals</p>
        """)
    
    # Main Search Section
    with gr.Column(elem_classes="search-card"):
        gr.HTML('<h3 class="section-header">🔍 What would you like to cook today?</h3>')
        
        user_query = gr.Textbox(
            label="Search Recipes",
            placeholder="e.g., 'healthy chicken salad', 'low-carb dinner', 'vegetarian pasta'...",
            lines=2,
            elem_classes="search-input"
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                dietary_choice = gr.Dropdown(
                    label="🥗 Dietary Preference",
                    choices=["None", "Vegetarian", "Vegan", "Gluten-Free"],
                    value="None",
                    elem_classes="dietary-dropdown"
                )
            
            with gr.Column(scale=2):
                health_condition = gr.Dropdown(
                    label="❤️ Health Focus",
                    choices=[
                        "None", "Diabetes", "Hypertension", "Heart-Friendly",
                        "Weight Loss", "Thyroid (Hypothyroidism)", "PCOS/PCOD",
                        "Kidney-Friendly", "Liver Health", "Anemia",
                        "Bone Health (Calcium & Vitamin D)"
                    ],
                    value="None",
                    elem_classes="health-dropdown"
                )
            
            with gr.Column(scale=3):
                allergen_input = gr.Textbox(
                    label="🚫 Exclude Allergens",
                    placeholder="e.g., nuts, dairy, eggs (comma-separated)",
                    elem_classes="allergen-input"
                )

    # Nutritional Goals Section
    with gr.Column(elem_classes="nutrition-goals"):
        gr.HTML('<h3>⚖️ Nutritional Goals (Optional - Per Meal)</h3>')
        gr.HTML('<p style="margin-bottom: 1rem; opacity: 0.9;">Set your target nutrition values to get more precise recommendations</p>')
        
        with gr.Row():
            max_calories = gr.Number(
                label="🔥 Max Calories",
                value=0,
                minimum=0,
                maximum=2000,
                step=50,
                elem_classes="nutrition-input"
            )
            min_protein = gr.Number(
                label="💪 Min Protein (g)",
                value=0,
                minimum=0,
                maximum=100,
                step=5,
                elem_classes="nutrition-input"
            )
            max_fat = gr.Number(
                label="🥑 Max Fat (g)",
                value=0,
                minimum=0,
                maximum=100,
                step=5,
                elem_classes="nutrition-input"
            )

    # Submit Button
    submit_btn = gr.Button(
        "🚀 Get My Personalized Recommendations",
        variant="primary",
        size="lg",
        elem_classes="submit-btn"
    )



    # Results Section
with gr.Column(elem_classes="results-section"):
    with gr.Column(elem_classes="results-card"):
        gr.HTML('<h3 class="section-header">📊 Results</h3>')
        
        latency_output = gr.HTML(
            label="Performance",
            elem_classes="latency-display"
        )
        
        # Keep only AI Recommendation tab
        with gr.Tabs():
            with gr.TabItem("🤖 AI Recommendation", elem_id="recommendation-tab"):
                rag_output = gr.Markdown(
                    label="Your Personalized Recommendation",
                    elem_classes="recommendation-output"
                )

# Connect the button to the function
submit_btn.click(
    fn=personalized_recipe,
    inputs=[user_query, dietary_choice, health_condition, allergen_input, max_calories, min_protein, max_fat],
    outputs=[latency_output, rag_output]  # Removed recipes_output
)

    
    
    
    # Footer
    gr.HTML("""
        <div style="text-align: center; margin-top: 3rem; padding: 2rem; color: rgba(255,255,255,0.8);">
            <p>✨ Powered by AI • Made with ❤️ for healthy living</p>
        </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)