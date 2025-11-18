# build_artifacts.py
import pickle
import os
import sys

# Ensure we can import from src
sys.path.append(os.getcwd())

from src.preprocess import preprocess_recipes
from src.embeddings import get_embeddings

DATA_PATH = "data/recipes.csv"
OUTPUT_FILE = "data/nutrisense_data.pkl"

def main():
    print("🚀 Starting local build process...")

    # 1. Preprocess Data
    print(f"📂 Reading and preprocessing {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found!")
        return
    
    df = preprocess_recipes(DATA_PATH)
    print(f"✅ Data loaded: {len(df)} recipes.")

    # 2. Generate Embeddings
    print("🧠 Generating embeddings (this may take a while)...")
    texts = df['chunk'].tolist()
    embeddings = get_embeddings(texts)

    # 3. Save Data + Embeddings (NOT the collection object)
    print(f"💾 Saving artifacts to {OUTPUT_FILE}...")
    
    artifacts = {
        "df": df,
        "embeddings": embeddings # <--- saving raw list/array instead of complex object
    }

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(artifacts, f)

    print("\n🎉 SUCCESS! Build complete.")
    print(f"👉 ACTION REQUIRED: Push '{OUTPUT_FILE}' to GitHub so Render can see it.")

if __name__ == "__main__":
    main()