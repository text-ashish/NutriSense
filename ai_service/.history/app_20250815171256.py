import gradio as gr
import time
from src.preprocess import preprocess_recipes
from src.embeddings import get_embeddings
from src.vectorstore import build_vectorstore, retrieve, collection
from src.rag import generate_response

DATA_PATH = "data/recipes.csv"

# --- Custom CSS for better styling ---
custom_css = """
.app-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
}

.app-header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.app-header p {
    margin: 0.5rem 0 0 0;
    font-size: 1.1rem;
    opacity: 0.9;
}

.search-section {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #667eea;
    margin: 1rem 0;
}

.nutrition-section {
    background: #fff3cd;
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #ffc107;
    margin: 1rem 0;
}

.results-section {
    background: #d1ecf1;
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #17a2b8;
    margin: 1rem 0;
}

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
}

.recipe-card {
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.btn-primary {
    background: linear-gradient(45deg, #667eea, #764ba2) !important;
    border: none !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    border-radius: 25px !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    transition: all 0.3s ease !important;
}

.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
}

.status-success {
    color: #28a745;
    font-weight: 600;
}

.status-warning {
    color: #ffc107;
    font-weight: 600;
}

.status-error {
    color: #dc3545;
    font-weight: 600;
}

/* Responsive design */
@media (max-width: 768px) {
    .app-header h1 {
        font-size: 2rem;
    }
    
    .search-section, .nutrition-section, .results-section {
        padding: 1rem;
    }
}
"""

# --- Enhanced data preparation with error handling ---
def prepare():
    try:
        print("🔄 Loading and preprocessing recipes...")
        df = preprocess_recipes(DATA_PATH)
        texts = df['chunk'].tolist()
        
        print("🔄 Generating embeddings...")
        embeddings = get_embeddings(texts)

        try:
            count = collection.count()
            print(f"📊 Found {count} recipes in vectorstore")
        except Exception as e:
            print(f"⚠️ Vectorstore error: {e}")
            count = 0
            
        if count == 0:
            print("🔄 Building vectorstore...")
            build_vectorstore(df, embeddings)
            print("✅ Vectorstore built successfully!")
            
        print(f"✅ Ready! Loaded {len(df)} recipes")
        return df, "success"
    except Exception as e:
        print(f"❌ Error during preparation: {e}")
        return None, f"Error: {str(e)}"

# Initialize the app
df, prep_status = prepare()

# --- Enhanced main function with better error handling ---
def personalized_recipe(
    user_query,
    dietary_choice,
    health_condition,
    allergen_input,
    max_calories,
    min_protein,
    max_fat,
    num_results
):
    # Input validation
    if not user_query or user_query.strip() == "":
        return (
            "⚠️ Please enter a recipe name, ingredient, or description to search.",
            "**No search query provided.** Please enter what you're looking for above.",
            [],
            "",
            "🔍 **Ready to search!** Enter a query above to get started."
        )
    
    if df is None:
        return (
            "❌ System Error",
            "**App initialization failed.** Please refresh the page and try again.",
            [],
            prep_status,
            "❌ **System Error:** Unable to load recipe database."
        )

    try:
        # Process inputs
        exclude_allergens = [a.strip().lower() for a in allergen_input.split(",") if a.strip()] if allergen_input else []
        
        # Validate nutritional goals
        nutrition_goals = []
        if max_calories > 0:
            nutrition_goals.append(f"Max {max_calories} calories")
        if min_protein > 0:
            nutrition_goals.append(f"Min {min_protein}g protein")
        if max_fat > 0:
            nutrition_goals.append(f"Max {max_fat}g fat")

        start = time.time()
        
        # Get query embedding
        print(f"🔍 Searching for: '{user_query}'")
        q_emb = get_embeddings([user_query.lower()])[0]

        # Retrieve recipes
        results = retrieve(
            query_embedding=q_emb,
            k=max(6, num_results),
            dietary_filter=dietary_choice if dietary_choice != "None" else None,
            exclude_allergens=exclude_allergens if exclude_allergens else None,
            health_condition=health_condition if health_condition != "None" else None
        )

        if not results:
            end = time.time()
            return (
                f"⏱ Search completed in {end-start:.2f}s",
                "**No recipes found** matching your criteria. Try:\n- Using different keywords\n- Relaxing dietary restrictions\n- Checking for typos",
                [],
                "No results found",
                "🔍 **No matches found.** Try adjusting your search terms or filters."
            )

        # Generate enhanced response via RAG
        answer = generate_response(
            user_query,
            retrieved_metadatas=results[:num_results],
            dietary_preference=dietary_choice if dietary_choice != "None" else None,
            exclude_allergens=exclude_allergens if exclude_allergens else None,
            health_condition=health_condition if health_condition != "None" else None,
            calorie_goal=max_calories if max_calories > 0 else None,
            protein_goal=min_protein if min_protein > 0 else None,
            fat_goal=max_fat if max_fat > 0 else None
        )

        end = time.time()
        latency = f"⏱ Found {len(results)} recipes in {end-start:.2f}s"

        # Format recipes for display
        formatted_recipes = []
        for i, r in enumerate(results[:num_results], 1):
            # Extract nutrition info safely
            nutrition = r.get('nutrition_normalized', {})
            calories = nutrition.get('calories', 'N/A')
            protein = nutrition.get('protein', 'N/A')
            fat = nutrition.get('fat', 'N/A')
            
            formatted_recipes.append({
                "Rank": f"#{i}",
                "Recipe Name": r.get('recipe_name', 'Unnamed Recipe').title(),
                "Key Ingredients": r.get('ingredients', 'Not specified')[:100] + "..." if len(r.get('ingredients', '')) > 100 else r.get('ingredients', 'Not specified'),
                "Calories": f"{calories}" + (" kcal" if calories != 'N/A' else ""),
                "Protein": f"{protein}" + ("g" if protein != 'N/A' else ""),
                "Fat": f"{fat}" + ("g" if fat != 'N/A' else ""),
                "Dietary": r.get('dietary_labels', 'None'),
                "Health Tags": r.get('health_tags', 'General')
            })

        # Create summary
        summary_parts = []
        if dietary_choice != "None":
            summary_parts.append(f"🥗 **{dietary_choice}** recipes")
        if health_condition != "None":
            summary_parts.append(f"💚 **{health_condition}** friendly")
        if exclude_allergens:
            summary_parts.append(f"🚫 Avoiding: **{', '.join(exclude_allergens)}**")
        if nutrition_goals:
            summary_parts.append(f"🎯 Goals: **{', '.join(nutrition_goals)}**")
            
        search_summary = f"🔍 **Search Results for '{user_query}'**\n" + (" | ".join(summary_parts) if summary_parts else "General search")

        status_msg = f"✅ **Success!** Found {len(results)} matching recipes"
        
        return latency, answer, formatted_recipes, status_msg, search_summary

    except Exception as e:
        print(f"❌ Error in personalized_recipe: {e}")
        return (
            "❌ Processing Error",
            f"**An error occurred:** {str(e)}\n\nPlease try again or contact support if the issue persists.",
            [],
            f"Error: {str(e)}",
            "❌ **Processing failed.** Please try again."
        )

# --- Enhanced Gradio interface ---
def create_interface():
    with gr.Blocks(css=custom_css, title="🍽 NutriSense — AI-Powered Recipe Assistant", theme=gr.themes.Soft()) as demo:
        
        # Header
        gr.HTML("""
        <div class="app-header">
            <h1>🍽 NutriSense</h1>
            <p>AI-Powered Personalized Recipe & Nutrition Assistant</p>
        </div>
        """)
        
        # Status indicator
        with gr.Row():
            status_display = gr.Markdown("🔍 **Ready to search!** Enter your preferences below.", elem_classes=["status-success"])
        
        # Main search section
        gr.HTML('<div class="search-section">')
        gr.Markdown("### 🔍 **Search & Preferences**")
        
        with gr.Row():
            with gr.Column(scale=2):
                user_query = gr.Textbox(
                    label="What are you looking for?",
                    placeholder="e.g., 'chicken curry', 'high protein breakfast', 'chocolate dessert'",
                    lines=2
                )
            with gr.Column(scale=1):
                num_results = gr.Slider(
                    label="Number of Results",
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1
                )
        
        with gr.Row():
            dietary_choice = gr.Dropdown(
                label="🥗 Dietary Preference",
                choices=["None", "Vegetarian", "Vegan", "Gluten-Free"],
                value="None"
            )
            health_condition = gr.Dropdown(
                label="💚 Health Condition",
                choices=[
                    "None", "Diabetes", "Hypertension", "Heart-Friendly",
                    "Weight Loss", "Thyroid (Hypothyroidism)", "PCOS/PCOD",
                    "Kidney-Friendly", "Liver Health", "Anemia",
                    "Bone Health (Calcium & Vitamin D)"
                ],
                value="None"
            )
            allergen_input = gr.Textbox(
                label="🚫 Avoid Allergens",
                placeholder="e.g., nuts, dairy, soy (comma-separated)"
            )
        gr.HTML('</div>')

        # Nutritional goals section
        gr.HTML('<div class="nutrition-section">')
        gr.Markdown("### 🎯 **Nutritional Goals** (Optional - per serving)")
        with gr.Row():
            max_calories = gr.Number(
                label="🔥 Max Calories",
                value=0,
                minimum=0,
                maximum=2000,
                info="0 = no limit"
            )
            min_protein = gr.Number(
                label="💪 Min Protein (g)",
                value=0,
                minimum=0,
                maximum=100,
                info="0 = no minimum"
            )
            max_fat = gr.Number(
                label="🧈 Max Fat (g)",
                value=0,
                minimum=0,
                maximum=100,
                info="0 = no limit"
            )
        gr.HTML('</div>')

        # Search button
        with gr.Row():
            submit_btn = gr.Button(
                "🔍 Find Personalized Recipes",
                variant="primary",
                elem_classes=["btn-primary"],
                size="lg"
            )

        # Results section
        gr.HTML('<div class="results-section">')
        gr.Markdown("### 📊 **Results**")
        
        with gr.Row():
            with gr.Column(scale=1):
                latency_output = gr.Textbox(label="⏱ Performance", interactive=False)
            with gr.Column(scale=1):
                system_status = gr.Textbox(label="📊 Status", interactive=False)
        
        search_summary = gr.Markdown("🔍 **Ready to search!** Enter your query above.")
        rag_output = gr.Markdown(label="🤖 **AI Recommendations**")
        
        recipes_output = gr.Dataframe(
            headers=["Rank", "Recipe Name", "Key Ingredients", "Calories", "Protein", "Fat", "Dietary", "Health Tags"],
            label="📋 **Matching Recipes**",
            interactive=False,
            wrap=True,
            max_height=400
        )
        gr.HTML('</div>')
        
        # Examples section
        gr.Markdown("### 💡 **Try These Examples**")
        examples = gr.Examples(
            examples=[
                ["high protein breakfast", "None", "Weight Loss", "", 400, 20, 15, 3],
                ["chocolate dessert", "Vegan", "None", "dairy, eggs", 300, 0, 10, 2],
                ["heart healthy dinner", "None", "Heart-Friendly", "", 500, 25, 15, 4],
                ["quick lunch under 30 minutes", "Vegetarian", "None", "", 350, 15, 0, 3],
                ["gluten free pasta", "Gluten-Free", "None", "", 0, 0, 0, 5]
            ],
            inputs=[user_query, dietary_choice, health_condition, allergen_input, max_calories, min_protein, max_fat, num_results],
            cache_examples=False
        )

        # Connect the button
        submit_btn.click(
            fn=personalized_recipe,
            inputs=[user_query, dietary_choice, health_condition, allergen_input, max_calories, min_protein, max_fat, num_results],
            outputs=[latency_output, rag_output, recipes_output, system_status, search_summary]
        )
        
        # Auto-clear status on new input
        user_query.change(
            lambda x: "🔍 **Ready to search!** Click the button below when ready." if x.strip() else "🔍 **Enter a search query** to get started.",
            inputs=[user_query],
            outputs=[status_display]
        )
        
        # Footer
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #666; margin-top: 2rem;">
            <p>🍽 <strong>NutriSense</strong> — Powered by AI for personalized nutrition recommendations</p>
            <p><em>Always consult with healthcare professionals for medical dietary advice</em></p>
        </div>
        """)
    
    return demo

# Create and configure the interface
demo = create_interface()

if __name__ == "__main__":
    print("🚀 Starting NutriSense...")
    print(f"📊 System Status: {prep_status}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None,
        show_api=False
    )