# src/personalize.py
import google.generativeai as genai
import os
import json

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

def personalize_recipe(recipe_meta, goals):
    """
    recipe_meta: dict from retrieve()
    goals: dict with keys like calorie_goal, carb_goal, dietary_filter, health_condition
    """
    prompt = f"""
You are a nutritionist and chef.
Take this recipe and rewrite it to match the user's health goals, dietary restrictions, and allergens.
Return it in this exact format:

🍽 Personalized Recipe for <Recipe Name> (<Dietary Tags>)

**Adjusted Recipe:**
- List main changes made

**Ingredients:**
- List with quantities

**Nutrition per serving:**
Calories: X kcal
Protein: Xg
Fat: Xg
Carbs: Xg
Sugar: Xg

**Why this recipe?**
- Short bullet points why it matches the user’s goals

**Preparation Steps:**
1. Step
2. Step
...

**Suggested Substitutions:**
- Substitution options

Here is the original recipe data (JSON):
{json.dumps(recipe_meta, indent=2)}

User goals & restrictions:
{json.dumps(goals, indent=2)}
"""
    response = model.generate_content(prompt)
    return response.text
