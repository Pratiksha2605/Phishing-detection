import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
json_file_path = "Phishing_data.json"

try:
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Invalid JSON format. Expected a list of dictionaries.")

    df = pd.DataFrame(data)

    if "text" not in df.columns:
        raise ValueError("Missing 'text' column in dataset.")

    df = df.dropna(subset=["text"])

    print("✅ Dataset Loaded Successfully!")
    print(f"Total Samples: {len(df)}")

    # Train the vectorizer with a fixed number of features
    vectorizer = TfidfVectorizer(max_features=5000)  # Ensure consistency
    X_text_tfidf = vectorizer.fit_transform(df["text"].astype(str))

    # Save the vectorizer
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print(f"✅ TF-IDF vectorizer saved successfully! Feature size: {X_text_tfidf.shape[1]}")

except FileNotFoundError:
    print(f"❌ Error: File '{json_file_path}' not found.")
except json.JSONDecodeError:
    print("❌ Error: Failed to decode JSON. Ensure it is correctly formatted.")
except ValueError as ve:
    print(f"❌ Data Error: {ve}")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
