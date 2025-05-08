import json
import joblib
import logging
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load dataset
json_file_path = "Phishing_data.json"

try:
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Ensure JSON is a list of dictionaries
    if not isinstance(data, list):
        raise ValueError("Invalid JSON format. Expected a list of dictionaries.")

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Ensure required columns exist
    required_columns = {"text", "label", "url"}  # Check if 'url' column exists
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Found columns: {df.columns}")

    # Drop empty text rows
    df = df.dropna(subset=["text"])

    # Show dataset info
    logger.info("✅ Dataset Loaded Successfully!")
    logger.info(f"Total Samples: {len(df)}")

    # Train TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)  # Ensure consistency
    X_text_tfidf = vectorizer.fit_transform(df["text"].astype(str))

    # Save the vectorizer
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    logger.info(f"✅ TF-IDF vectorizer saved! Feature size: {X_text_tfidf.shape[1]}")

    # Extract URL features
    def extract_url_features(url):
        parsed = urlparse(url)
        return pd.DataFrame([{
            "url_length": len(url),
            "num_dots": url.count("."),
            "num_hyphens": url.count("-"),
            "num_slashes": url.count("/"),
            "num_question_marks": url.count("?"),
            "num_equals": url.count("="),
            "num_at": url.count("@"),
            "domain_length": len(parsed.netloc),
            "path_length": len(parsed.path),
            "is_ip": 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", parsed.netloc) else 0,
        }])

    # Apply URL feature extraction
    url_features = pd.DataFrame([extract_url_features(url) for url in df["url"]])

    # Combine TF-IDF features with URL features
    X = pd.concat([pd.DataFrame(X_text_tfidf.toarray()), url_features], axis=1)

    # Prepare data for XGBoost
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Train XGBoost model
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5
    }
    num_boost_round = 100  # ✅ Corrected parameter

    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

    # Save the trained model
    model.save_model("xgboost_model.json")
    logger.info("✅ XGBoost Model trained and saved successfully as JSON!")

except FileNotFoundError:
    logger.error(f"❌ Error: File '{json_file_path}' not found.")
except json.JSONDecodeError:
    logger.error("❌ Error: Failed to decode JSON. Ensure it is correctly formatted.")
except ValueError as ve:
    logger.error(f"❌ Data Error: {ve}")
except Exception as e:
    logger.error(f"❌ An unexpected error occurred: {e}")
