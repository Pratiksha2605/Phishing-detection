import json
import pandas as pd
import re
import numpy as np
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import joblib
from tqdm import tqdm

tqdm.pandas()  # Enable progress bar for pandas apply

# === STEP 1: Load Dataset ===
json_file_path = r"C:\Users\Pratiksha Yadav\Desktop\PROJECT\combined_reduced.json"

with open(json_file_path, "r", encoding="utf-8") as file:
    data = json.load(file)
df = pd.DataFrame(data)
print(f"✅ Dataset loaded: {df.shape}")

# === STEP 2: Classify Input Type ===
def classify_type(text):
    if isinstance(text, str):
        if text.startswith("http") or "www." in text:
            return "URL"
        elif any(tag in text.lower() for tag in ["<html", "<script", "<form", "<iframe"]):
            return "HTML"
        elif "@" in text and "." in text.split("@")[1]:
            return "EMAIL"
        else:
            return "TEXT"
    return "UNKNOWN"

df["type"] = df["text"].apply(classify_type)

# === STEP 3: Feature Engineering for URL ===
def extract_url_features(url):
    try:
        parsed = urlparse(url)
        return [
            len(url),
            len(parsed.netloc),
            url.count('.'),
            url.count('-'),
            url.count('/'),
            1 if re.match(r"\d+\.\d+\.\d+\.\d+", parsed.netloc) else 0
        ]
    except:
        return [0] * 6

# === STEP 4: Get DistilBERT Embeddings ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

def get_embedding(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    except:
        return np.zeros(768)

# === STEP 5: Combine Features ===
print("🔍 Extracting DistilBERT embeddings...")
df["embedding"] = df["text"].progress_apply(get_embedding)

print("🔍 Extracting URL features...")
df["url_features"] = df.apply(lambda row: extract_url_features(row["text"]) if row["type"] == "URL" else [0]*6, axis=1)

# Final feature vector
X = np.hstack([
    np.vstack(df["embedding"].values),
    np.vstack(df["url_features"].values)
])

y = df["label"].astype(int).values

# === STEP 6: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === STEP 7: Model Training ===
print("🚀 Training models...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, 
                    tree_method="gpu_hist" if torch.cuda.is_available() else "hist")

ensemble = VotingClassifier(estimators=[("rf", rf), ("xgb", xgb)], voting="soft")
ensemble.fit(X_train, y_train)

# === STEP 8: Evaluation ===
y_pred = ensemble.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc:.4f}")

# === STEP 9: Save the Model ===
model_path = r"C:\Users\Pratiksha Yadav\Desktop\PROJECT\ensemble_phishing_model.pkl"
joblib.dump(ensemble, model_path)
print(f"✅ Model saved to {model_path}")
