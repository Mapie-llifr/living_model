# visualize.py

import torch
from model import MLP
from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

MODEL_PATH = "checkpoints/model.pt"
TEST_WORDS_PATH = "data/test_words.txt"
SAVE_DIR = "visualizations"

os.makedirs(SAVE_DIR, exist_ok=True)

# Load words
with open(TEST_WORDS_PATH, "r") as f:
    words = [line.strip() for line in f.readlines()]

# Load embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(words)

X = torch.tensor(embeddings, dtype=torch.float32)

# Load model
model = MLP()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Get predictions
with torch.no_grad():
    preds = model(X).numpy().flatten()

# UMAP projection
reducer = umap.UMAP(n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(embeddings)

# Plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    embedding_2d[:, 0],
    embedding_2d[:, 1],
    c=preds,
    cmap="coolwarm",
    alpha=0.8
)

plt.colorbar(scatter, label="Predicted Probability")
plt.title("Decision Boundary Visualization")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(SAVE_DIR, f"viz_{timestamp}.png")
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Visualization saved at {save_path}")

