# live_train.py

import torch
from model import MLP
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch.optim as optim
import os
import csv
from datetime import datetime
import pandas as pd

# --------------------
# Paths
# --------------------

MODEL_PATH = "checkpoints/model.pt"
LOG_PATH = "logs/training_log.csv"

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# --------------------
# Init embedder & model
# --------------------

embedder = SentenceTransformer("all-MiniLM-L6-v2")
model = MLP()

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Loaded existing model.")
else:
    print("Initialized new model.")

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

# --------------------
# Init memory from log
# --------------------

memory = []
max_memory = 100

if os.path.exists(LOG_PATH):
    df = pd.read_csv(LOG_PATH)
    for _, row in df.iterrows():
        emb = embedder.encode([row["word"]])
        x_mem = torch.tensor(emb, dtype=torch.float32)
        y_mem = torch.tensor([[float(row["feedback"])]], dtype=torch.float32)
        memory.append((x_mem, y_mem))

# --------------------
# Init log file header
# --------------------

if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "word", "feedback"])

# --------------------
# Live loop
# --------------------

new_weight = 3
total_updates = 0

print("Live training started. Type 'quit' to stop.\n")

while True:
    word = input("Enter a word: ")

    if word == "quit":
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

    # Add to memory
    memory.append((x.detach(), y.detach()))
    if len(memory) > max_memory:
        memory.pop(0)

    # Build batch
    batch_x = torch.cat(
        [item[0] for item in memory] +
        [x.repeat(new_weight, 1)]
    )
    batch_y = torch.cat(
        [item[1] for item in memory] +
        [y.repeat(new_weight, 1)]
    )

    preds = model(batch_x)
    loss = loss_fn(preds, batch_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Monitoring
    with torch.no_grad():
        weight_norm = sum(p.norm().item() for p in model.parameters())
        mean_pred = model(batch_x).mean().item()

    total_updates += 1
    torch.save(model.state_dict(), MODEL_PATH)

    print(f"Weight norm: {weight_norm:.2f}")
    print(f"Mean prediction over memory: {mean_pred:.3f}")
    print(f"Total updates: {total_updates}")
    print(f"Updated model | Loss: {loss.item():.4f}\n")

    # Log
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), word, feedback])
