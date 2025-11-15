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

# --- Gradio interface ---
with gr.Blocks(title="🍽 NutriSense — Personalized Recipe & Nutrition RAG") as demo:
    gr.Markdown(
        """
        # 🍽 NutriSense
        **Personalized Recipe & Nutrition Assistant**

        Follow the steps below to get recipe recommendations tailored to your dietary and health preferences.
        """
    )

    # --- Step 1: Recipe query ---
    with gr.Group():
        gr.Markdown("### 1️⃣ Search for a Recipe")
        user_query = gr.Textbox(
            label="Recipe Name, Ingredient, or Description",
            placeholder="e.g., chocolate cake, chicken salad",
            lines=1
        )

    # --- Step 2: Dietary & Health filters ---
    with gr.Accordion("2️⃣ Dietary & Health Filters (Optional)", open=False):
        dietary_choice = gr.Dropdown(
            label="Dietary Preference",
            choices=["None", "Vegetarian", "Vegan", "Gluten-Free"],
            value="None",
            info="Select your diet type to filter recipes accordingly."
        )
        health_condition = gr.Dropdown(
            label="Health Condition",
            choices=[
                "None", "Diabetes", "Hypertension", "Heart-Friendly",
                "Weight Loss", "Thyroid (Hypothyroidism)", "PCOS/PCOD",
                "Kidney-Friendly", "Liver Health", "Anemia",
                "Bone Health (Calcium & Vitamin D)"
            ],
            value="None",
            info="Select a health condition to tailor recipes."
        )
        allergen_input = gr.Textbox(
            label="Exclude Allergens",
            placeholder="e.g., peanuts, gluten, dairy",
            info="Comma-separated list of ingredients you want to avoid."
        )

    # --- Step 3: Nutritional Goals ---
    with gr.Accordion("3️⃣ Nutritional Goals (Optional)", open=False):
        max_calories = gr.Number(label="Max Calories per meal", value=0, info="0 means no limit")
        min_protein = gr.Number(label="Min Protein (g)", value=0, info="0 means no minimum")
        max_fat = gr.Number(label="Max Fat (g)", value=0, info="0 means no limit")

    # --- Step 4: Outputs ---
    latency_output = gr.Textbox(label="⏱ Latency", interactive=False)
    rag_output = gr.Markdown(label="💡 Personalized Response")
    recipes_output = gr.Dataframe(
        headers=["Recipe Name","Ingredients","Directions","Nutrition","Dietary labels","Allergens","Substitutions","Health tags"],
        label="📜 Retrieved Recipes"
    )

    # --- Step 5: Submit ---
    submit_btn = gr.Button("Get Personalized Recommendation")
    submit_btn.click(
        fn=personalized_recipe,
        inputs=[user_query, dietary_choice, health_condition, allergen_input, max_calories, min_protein, max_fat],
        outputs=[latency_output, rag_output, recipes_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
