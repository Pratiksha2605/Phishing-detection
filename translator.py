import json
import joblib
import re
import xgboost as xgb
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
from googletrans import Translator  # Import Translator for English to Hindi translation

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)  # Allow CORS for frontend integration

# Initialize translator
translator = Translator()

# Load the XGBoost model
try:
    model = xgb.Booster()
    model.load_model("xgboost_model.json")
    logger.info("✅ XGBoost model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")

# Load the TF-IDF vectorizer
try:
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    expected_feature_count = 5000  # Ensure consistency with training
    logger.info(f"✅ TF-IDF vectorizer loaded with {expected_feature_count} features!")
except Exception as e:
    logger.error(f"❌ Error loading vectorizer: {e}")
    vectorizer = None

def translate_to_hindi(text):
    """Translate English text to Hindi."""
    try:
        translated_text = translator.translate(text, src="en", dest="hi").text
        return translated_text
    except Exception as e:
        logger.error(f"⚠️ Translation failed: {e}")
        return text  # Return original text if translation fails

def preprocess_text(text):
    """Preprocess text for model prediction."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    try:
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words("english")]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)
    except Exception as e:
        logger.error(f"⚠️ Text preprocessing failed: {e}")
        return ""

def extract_url_features(url):
    """Extract features from URL."""
    try:
        parsed_url = urlparse(url)
        features = {
            "url_length": len(url),
            "num_dots": url.count("."),
            "num_hyphens": url.count("-"),
            "num_slashes": url.count("/"),
            "num_question_marks": url.count("?"),
            "num_equals": url.count("="),
            "num_at": url.count("@"),
            "domain_length": len(parsed_url.netloc),
            "path_length": len(parsed_url.path),
            "is_ip": 1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", parsed_url.netloc) else 0,
        }
        return pd.DataFrame([features])
    except Exception as e:
        logger.error(f"⚠️ URL Feature Extraction Failed: {e}")
        return pd.DataFrame([{key: 0 for key in features}])

@app.route("/predict", methods=["POST"])
def predict():
    """Handle phishing prediction requests."""
    try:
        data = request.get_json()
        if not data or "text" not in data or "url" not in data:
            logger.error("❌ Missing 'text' or 'url' in request")
            return jsonify({"error": "Missing 'text' or 'url' in request"}), 400

        email_text = data.get("text", "")
        email_url = data.get("url", "")

        # Translate email text to Hindi
        email_text_hindi = translate_to_hindi(email_text)

        # Preprocess translated text
        cleaned_text = preprocess_text(email_text_hindi)
        url_features = extract_url_features(email_url)

        if vectorizer is None:
            return jsonify({"error": "TF-IDF vectorizer not loaded!"}), 500

        features_tfidf = vectorizer.transform([cleaned_text])
        if features_tfidf.shape[1] != expected_feature_count:
            logger.error(f"❌ Feature mismatch! Expected {expected_feature_count}, got {features_tfidf.shape[1]}")
            return jsonify({"error": "Feature mismatch between vectorizer and model!"}), 500

        full_features = np.hstack((features_tfidf.toarray(), url_features.to_numpy()))
        dmatrix = xgb.DMatrix(full_features)
        prediction = model.predict(dmatrix)
        predicted_label = int(prediction[0] >= 0.5)

        # Translate prediction result to Hindi
        hindi_result = "यह एक फ़िशिंग ईमेल है।" if predicted_label == 1 else "यह एक सुरक्षित ईमेल है।"
        logger.info(f"✅ Prediction: {predicted_label} ({hindi_result})")

        return jsonify({"prediction": predicted_label, "message": hindi_result})
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=False)
