# train.py

import torch
from dataset import load_dataset
from model import MLP
from sentence_transformers import SentenceTransformer

texts, labels = load_dataset("data/words.csv")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
X = embedder.encode(texts)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.BCELoss()

for epoch in range(200):
    preds = model(X)
    loss = loss_fn(preds, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss {loss.item():.4f}")
