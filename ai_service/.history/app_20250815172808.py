with gr.Blocks(title="🍽 NutriSense — Personalized Recipe & Nutrition RAG", theme="default") as demo:
    gr.Markdown("## 🍽 NutriSense — Personalized Recipe & Nutrition RAG")
    gr.Markdown("Find recipes tailored to your dietary preferences, health conditions, and nutritional goals.")

    with gr.Tab("Search & Preferences"):
        with gr.Row():
            user_query = gr.Textbox(
                label="Search Recipes",
                placeholder="Enter recipe name, ingredient, or description...",
                info="Type what you're craving or ingredients you have."
            )
        with gr.Row():
            dietary_choice = gr.Dropdown(
                label="Dietary Preference",
                choices=["None", "Vegetarian", "Vegan", "Gluten-Free"],
                value="None"
            )
            health_condition = gr.Dropdown(
                label="Health Condition",
                choices=[
                    "None", "Diabetes", "Hypertension", "Heart-Friendly",
                    "Weight Loss", "Thyroid (Hypothyroidism)", "PCOS/PCOD",
                    "Kidney-Friendly", "Liver Health", "Anemia",
                    "Bone Health (Calcium & Vitamin D)"
                ],
                value="None"
            )
        with gr.Row():
            allergen_input = gr.Textbox(
                label="Exclude Allergens",
                placeholder="e.g., peanuts, shellfish",
                info="Separate multiple allergens with commas."
            )

    with gr.Tab("Nutritional Goals (optional)"):
        with gr.Row():
            max_calories = gr.Number(label="Max Calories per meal", value=0, info="Set 0 to ignore")
            min_protein = gr.Number(label="Min Protein (g)", value=0, info="Set 0 to ignore")
            max_fat = gr.Number(label="Max Fat (g)", value=0, info="Set 0 to ignore")

    with gr.Tab("Results"):
        latency_output = gr.Textbox(label="⏱ Latency")
        rag_output = gr.Markdown(label="💡 Personalized Recommendation")
        recipes_output = gr.Dataframe(
            headers=["Recipe Name","Ingredients","Directions","Nutrition","Dietary labels","Allergens","Substitutions","Health tags"],
            label="📋 Retrieved Recipes",
            interactive=False,
            max_rows=10
        )

    submit_btn = gr.Button("Get Personalized Recommendation", variant="primary")
    submit_btn.click(
        fn=personalized_recipe,
        inputs=[user_query, dietary_choice, health_condition, allergen_input, max_calories, min_protein, max_fat],
        outputs=[latency_output, rag_output, recipes_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
