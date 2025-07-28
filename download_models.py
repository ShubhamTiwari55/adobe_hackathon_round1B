# download_models.py
from sentence_transformers import SentenceTransformer, CrossEncoder
import os

# Create the models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

print("Downloading SentenceTransformer: all-MiniLM-L6-v2...")
st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
st_model.save('./models/all-MiniLM-L6-v2')
print("...Done.")

print("Downloading CrossEncoder: ms-marco-MiniLM-L-6-v2...")
ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
ce_model.save('./models/ms-marco-MiniLM-L-6-v2')
print("...Done.")

print("\n✅ All models downloaded successfully into the 'models' folder.")