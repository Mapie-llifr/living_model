# live_train.py

import torch
from model import MLP
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch.optim as optim

# Init
embedder = SentenceTransformer("all-MiniLM-L6-v2")
model = MLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

print("Live training started. Type 'quit' to stop.\n")

while True:
    word = input("Enter a word: ")

    if word == "quit":
        total_updates += 1
        torch.save(model.state_dict(), "model_checkpoint.pt")
        break

    embedding = embedder.encode([word])
    x = torch.tensor(embedding, dtype=torch.float32)

    with torch.no_grad():
        pred = model(x)
        print(f"Model prediction (prob red): {pred.item():.3f}")

    feedback = input("Is this correct? (1=yes, 0=no): ")

    if feedback not in ["0", "1"]:
        print("Invalid input.\n")
        continue

    y = torch.tensor([[float(feedback)]], dtype=torch.float32)

    # Training step
    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Updated model | Loss: {loss.item():.4f}\n")
