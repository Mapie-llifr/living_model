# visualize.py

import umap
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from dataset import load_dataset

texts, labels = load_dataset("data/words.csv")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
X = embedder.encode(texts)

reducer = umap.UMAP()
X2d = reducer.fit_transform(X)

plt.scatter(X2d[:,0], X2d[:,1], c=labels)
plt.title("Semantic space (human labels)")
#plt.show()

plt.savefig("experiments/umap_epoch_01.png", dpi=300)
plt.close()

