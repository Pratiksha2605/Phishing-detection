from transformers import BertTokenizer
import joblib

# Define tokenizer path
tokenizer_path = r"ensemble_tokenizer.pkl"

# Load BERT tokenizer (same as used during training)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Save tokenizer using joblib
joblib.dump(tokenizer, tokenizer_path)

print(f"✅ Tokenizer saved successfully at {tokenizer_path}!")
