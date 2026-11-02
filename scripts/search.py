import json
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/index/legal_judgments.index"
METADATA_PATH = "data/embeddings/chunks_metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading metadata...")

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Search engine ready.\n")

while True:

    query = input("Enter legal query (or type 'exit'): ").strip()

    if query.lower() == "exit":
        print("Exiting...")
        break

    if not query:
        print("Please enter a query.\n")
        continue

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding,
        TOP_K
    )

    print("\n" + "=" * 70)
    print("SEARCH RESULTS")
    print("=" * 70)

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        result = metadata[idx]

        print(f"\nResult {rank}")
        print("-" * 70)

        print(f"Case ID: {result['document_id']}")
        print(f"Pages: {result['start_page']}-{result['end_page']}")
        print(f"Similarity: {score:.4f}")

        print("\nRelevant passage:")
        print(result["text"][:1000])

    print("\n" + "=" * 70 + "\n")