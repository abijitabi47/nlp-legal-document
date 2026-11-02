import os
import numpy as np
import faiss

EMBEDDINGS_PATH = "data/embeddings/embeddings.npy"
INDEX_DIR = "data/index"
INDEX_PATH = os.path.join(INDEX_DIR, "legal_judgments.index")

os.makedirs(INDEX_DIR, exist_ok=True)

print("Loading embeddings...")

embeddings = np.load(EMBEDDINGS_PATH)

print(f"Embedding shape: {embeddings.shape}")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

print("Adding embeddings to FAISS index...")

index.add(embeddings)

print(f"Total vectors in index: {index.ntotal}")

faiss.write_index(index, INDEX_PATH)

print("\nFAISS index created successfully.")
print(f"Saved to: {INDEX_PATH}")