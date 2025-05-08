from flask import Flask, request, jsonify
from flask_cors import CORS
import pymongo
import datetime
import re
import torch
import joblib
import numpy as np
from urllib.parse import urlparse
from transformers import BertTokenizer, BertModel
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

app = Flask(__name__)
CORS(app)

# === MongoDB Connection ===
MONGO_URI = "mongodb+srv://pencilemaniacs03:<db_password>@cluster0.wtc3h.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URI)
db = client["phishingDB"]
collection = db["detections"]

# === Load Hybrid Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

model_path = "ensemble_phishing_model.pkl"  # Ensure model path is correct
ensemble_model = joblib.load(model_path)

# === URL Feature Extraction ===
def extract_url_features(url):
    parsed_url = urlparse(url)
    contains_ip = 1 if re.match(r"\d+\.\d+\.\d+\.\d+", parsed_url.netloc) else 0
    return np.array([
        len(url), len(parsed_url.netloc), url.count('.'),
        url.count('-'), url.count('/'), contains_ip
    ]).reshape(1, -1)

# === Text Feature Extraction (BERT) ===
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten().reshape(1, -1)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    email_text = data.get("text", "")
    email_url = data.get("url", "")

    # === Process Text ===
    text_prediction = None
    if email_text:
        text_features = get_bert_embedding(email_text)
        text_prediction = ensemble_model.predict(text_features)[0]

    # === Process URL ===
    url_prediction = None
    if email_url:
        url_features = extract_url_features(email_url)
        url_prediction = ensemble_model.predict(url_features)[0]

    # Determine Final Prediction
    final_prediction = max(text_prediction or 0, url_prediction or 0)

    # === Store in MongoDB ===
    detection_log = {
        "timestamp": datetime.datetime.utcnow(),
        "email_text": email_text[:100] + "..." if len(email_text) > 100 else email_text,
        "url": email_url,
        "prediction": "Phishing" if final_prediction == 1 else "Safe"
    }
    collection.insert_one(detection_log)

    return jsonify({"prediction": final_prediction})

@app.route("/logs", methods=["GET"])
def get_logs():
    """Retrieve the last 10 phishing logs."""
    logs = list(collection.find().sort("timestamp", -1).limit(10))
    for log in logs:
        log["_id"] = str(log["_id"])  # Convert ObjectId to string
    return jsonify(logs)

if __name__ == "__main__":
    app.run(debug=False)
