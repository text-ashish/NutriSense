import pandas as pd
import re

def normalize_nutrition(nutrition_dict):
    """Convert nutrition info to standard units."""
    standard = {}
    standard['calories'] = float(nutrition_dict.get('calories', 0))
    standard['protein_g'] = float(str(nutrition_dict.get('protein','0').replace('g','')))
    standard['fat_g'] = float(str(nutrition_dict.get('fat','0').replace('g','')))
    standard['carbs_g'] = float(str(nutrition_dict.get('carbs','0').replace('g','')))
    standard['sugar_g'] = float(str(nutrition_dict.get('sugar','0').replace('g','')))
    standard['fiber_g'] = float(str(nutrition_dict.get('fiber','0').replace('g','')))
    standard['sodium_mg'] = float(str(nutrition_dict.get('sodium','0').replace('mg','')))
    # Vitamins example
    vitamins = nutrition_dict.get('vitamins', {})
    standard['vitamin_A_IU'] = float(str(vitamins.get('A','0').replace(' IU',''))) 
    standard['vitamin_C_mg'] = float(str(vitamins.get('C','0').replace(' mg','')))
    return standard

def preprocess_recipes(file_path):
    df = pd.read_csv(file_path)
    df['nutrition_normalized'] = df['nutrition'].apply(lambda x: normalize_nutrition(eval(x)))
    
    # Chunking: combine ingredients + directions + nutrition for embeddings
    df['chunk'] = df.apply(lambda row: f"Recipe: {row['recipe_name']}\nIngredients: {row['ingredients']}\nDirections: {row['directions']}\nNutrition: {row['nutrition_normalized']}\nDietary labels: {row['dietary_labels']}\nAllergens: {row['allergens']}", axis=1)
    
    return df
