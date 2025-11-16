# src/preprocess.py
import pandas as pd
import ast

def normalize_nutrition_field(nut):
    """
    Normalize nutrition dictionary to consistent numeric fields.
    Expects nut as dict or string-repr of dict.
    Returns standardized dict with keys: calories, protein_g, fat_g, carbs_g, sugar_g, fiber_g, sodium_mg, vitamins
    """
    if nut is None:
        return {}
    if isinstance(nut, str):
        try:
            nut = ast.literal_eval(nut)
        except Exception:
            return {}

    out = {}
    def to_float(x):
        try:
            if x is None:
                return 0.0
            s = str(x)
            s = s.replace("kcal","").replace("Kcal","").replace(" kcal","").strip()
            s = s.replace("g","").replace(" mg","").replace("mg","")
            return float(s)
        except:
            return 0.0

    out['calories'] = to_float(nut.get('calories', 0))
    out['protein_g'] = to_float(nut.get('protein', 0))
    out['fat_g'] = to_float(nut.get('fat', 0))
    out['carbs_g'] = to_float(nut.get('carbs', 0))
    out['sugar_g'] = to_float(nut.get('sugar', 0))
    out['fiber_g'] = to_float(nut.get('fiber', 0))
    # sodium: support "0.4 g" or "400 mg"
    sodium = nut.get('sodium', 0)
    try:
        s = str(sodium)
        if 'g' in s and 'mg' not in s:
            out['sodium_mg'] = float(s.replace('g','').strip()) * 1000
        elif 'mg' in s:
            out['sodium_mg'] = float(s.replace('mg','').strip())
        else:
            out['sodium_mg'] = float(s)
    except:
        out['sodium_mg'] = 0.0

    out['vitamins'] = nut.get('vitamins', {})
    return out

def preprocess_recipes(csv_path):
    """
    Load CSV and produce DataFrame with normalized nutrition and a text chunk for embeddings.
    Expected CSV columns:
    recipe_name, ingredients (string), directions, prep_time, cook_time, total_time, servings, nutrition, dietary_labels, allergens, substitutions, health_tags (optional)
    """
    df = pd.read_csv(csv_path)
    # Fill missing columns with defaults
    for col in ['ingredients','directions','prep_time','cook_time','total_time','servings','nutrition','dietary_labels','allergens','substitutions','health_tags']:
        if col not in df.columns:
            df[col] = [''] * len(df)

    # Normalize nutrition
    df['nutrition_normalized'] = df['nutrition'].apply(normalize_nutrition_field)

    # Ensure dietary_labels, allergens, health_tags are strings/lists
    def ensure_str(x):
        if pd.isna(x):
            return ''
        if isinstance(x, (list, tuple)):
            return ",".join(x)
        return str(x)

    df['dietary_labels'] = df['dietary_labels'].apply(ensure_str)
    df['allergens'] = df['allergens'].apply(ensure_str)
    df['health_tags'] = df['health_tags'].apply(ensure_str)

    # Build text chunk for embeddings
    def make_chunk(row):
        return (
            f"Recipe: {row['recipe_name']}\n"
            f"Ingredients: {row['ingredients']}\n"
            f"Directions: {row['directions']}\n"
            f"Nutrition: {row['nutrition_normalized']}\n"
            f"Dietary labels: {row['dietary_labels']}\n"
            f"Allergens: {row['allergens']}\n"
            f"Health tags: {row['health_tags']}\n"
            f"Substitutions: {row['substitutions']}"
        )

    df['chunk'] = df.apply(make_chunk, axis=1)

    return df
