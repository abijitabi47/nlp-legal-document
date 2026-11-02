
import json
import os
import re
import sys
import faiss
import numpy as np

from collections import defaultdict
from sentence_transformers import SentenceTransformer

# Allow importing category_detection.py from the scripts folder.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from category_detection import detect_categories


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INDEX_PATH = os.path.join(
    PROJECT_DIR, "data", "index", "legal_judgments.index"
)

METADATA_PATH = os.path.join(
    PROJECT_DIR, "data", "embeddings", "chunks_metadata.json"
)

MODEL_NAME = "all-MiniLM-L6-v2"

CANDIDATE_K = 30
TOP_CASES = 5
TOP_CHUNKS_PER_CASE = 3
PASSAGE_PREVIEW_LENGTH = 800


# --------------------------------------------------
# LOAD MODEL, INDEX AND METADATA
# --------------------------------------------------

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading chunk metadata...")

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

if index.ntotal != len(metadata):
    raise ValueError(
        f"Index contains {index.ntotal} vectors, "
        f"but metadata contains {len(metadata)} chunks."
    )

print(f"Search ready: {index.ntotal} indexed chunks.\n")


# --------------------------------------------------
# CATEGORY DETECTION HELPERS
# --------------------------------------------------

def get_query_categories(query):
    """Detect categories mentioned in the user's query."""
    return detect_categories(query)


def get_case_categories(matches):
    """
    Detect categories from the retrieved passages of a case.

    Categories are based on keyword matches, not a trained
    legal classifier or verified case-level labels.
    """
    combined_text = " ".join(
        match["chunk"].get("text", "")
        for match in matches
    )

    return detect_categories(combined_text)


# --------------------------------------------------
# SEARCH AND RANK DISTINCT JUDGMENTS
# --------------------------------------------------

def search_cases(query):
    """Retrieve similar chunks and group them by judgment."""

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    k = min(CANDIDATE_K, index.ntotal)

    scores, indices = index.search(query_embedding, k)

    grouped = defaultdict(list)

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0 or idx >= len(metadata):
            continue

        chunk = metadata[idx]
        document_id = chunk.get(
            "document_id",
            chunk.get("filename", f"unknown_{idx}")
        )

        grouped[document_id].append({
            "score": float(score),
            "chunk": chunk
        })

    ranked_cases = []

    for document_id, matches in grouped.items():

        matches.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        best_matches = matches[:TOP_CHUNKS_PER_CASE]

        case_score = sum(
            item["score"] for item in best_matches
        ) / len(best_matches)

        categories = get_case_categories(best_matches)

        ranked_cases.append({
            "document_id": document_id,
            "score": case_score,
            "matches": best_matches,
            "categories": categories
        })

    ranked_cases.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return ranked_cases[:TOP_CASES]


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

def display_categories(categories):

    if not categories:
        print("  - No category detected from retrieved passages")
        return

    for category, keywords in categories.items():
        print(f"  - {category}")
        print(f"    Matching terms: {', '.join(keywords)}")


def display_results(query, results):

    print("\n" + "=" * 75)
    print("LEGAL CASE ANALYSIS AND SIMILARITY SEARCH")
    print("=" * 75)

    print(f"\nQuery: {query}")

    query_categories = get_query_categories(query)

    print("\nDetected query categories:")
    display_categories(query_categories)

    print("\n" + "-" * 75)
    print("SIMILAR LEGAL CASES")
    print("-" * 75)

    if not results:
        print("No matching cases found.")
        return

    for rank, case in enumerate(results, start=1):

        first_chunk = case["matches"][0]["chunk"]

        print(f"\nResult {rank}")
        print("-" * 75)

        print(f"Document ID: {case['document_id']}")
        print(
            "File:",
            first_chunk.get("filename", "Unknown")
        )
        print(f"Ranking score: {case['score']:.4f}")

        print("\nDetected case categories:")
        display_categories(case["categories"])

        print("\nRelevant passages:")

        for match in case["matches"]:

            chunk = match["chunk"]

            start_page = chunk.get("start_page", "?")
            end_page = chunk.get("end_page", "?")

            print(
                f"\nPages {start_page}-{end_page} "
                f"| Chunk similarity: {match['score']:.4f}"
            )

            passage = chunk.get("text", "").strip()

            if len(passage) > PASSAGE_PREVIEW_LENGTH:
                passage = (
                    passage[:PASSAGE_PREVIEW_LENGTH]
                    + "..."
                )

            print(passage if passage else "[No text available]")

    print("\n" + "=" * 75)

    print(
        "\nNote: Similarity scores measure vector similarity; "
        "they are not probabilities of legal relevance."
    )
    print(
        "Category labels are keyword-based and may be incomplete "
        "or incorrect. Verify the original judgment."
    )


# --------------------------------------------------
# INTERACTIVE SEARCH LOOP
# --------------------------------------------------

def main():

    print("Enter a legal case description or question.")
    print("Type 'exit' to quit.\n")

    while True:

        try:
            query = input("Enter your legal query: ").strip()

        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if query.lower() in {"exit", "quit"}:
            print("Exiting...")
            break

        if not query:
            print("Please enter a non-empty query.\n")
            continue

        try:
            results = search_cases(query)
            display_results(query, results)

        except Exception as error:
            print(f"\nSearch failed: {error}")

        print()


if __name__ == "__main__":
    main()
