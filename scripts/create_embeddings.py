import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_DIR = "data/chunks"
OUTPUT_DIR = "data/embeddings"

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

texts = []
metadata = []

files = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(".json")
]

print(f"Found {len(files)} chunk files.")

for filename in files:
    path = os.path.join(INPUT_DIR, filename)

    with open(path, "r", encoding="utf-8") as f:
        document = json.load(f)

    document_id = document["document_id"]

    for chunk in document["chunks"]:
        text = chunk["text"].strip()

        if not text:
            continue

        texts.append(text)

        metadata.append({
            "document_id": document_id,
            "filename": document["filename"],
            "chunk_id": chunk["chunk_id"],
            "start_page": chunk["start_page"],
            "end_page": chunk["end_page"],
            "word_count": chunk["word_count"],
            "text": text
        })

print(f"Total chunks: {len(texts)}")
print("Generating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.asarray(embeddings, dtype="float32")

np.save(
    os.path.join(OUTPUT_DIR, "embeddings.npy"),
    embeddings
)

with open(
    os.path.join(OUTPUT_DIR, "chunks_metadata.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("\nEmbedding generation complete.")
print(f"Embedding shape: {embeddings.shape}")
print(f"Saved: {OUTPUT_DIR}/embeddings.npy")
print(f"Saved: {OUTPUT_DIR}/chunks_metadata.json")