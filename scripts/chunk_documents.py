import os
import json
import re

INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/chunks"

CHUNK_SIZE = 500
OVERLAP = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_text(text):
    text = re.sub(r"\b[A-H](?:\s*\n\s*[A-H])+\b", " ", text)
    text = re.sub(r"SUPREME COURT REPORTS", " ", text)
    text = re.sub(r"\[\d{4}\]\s+\d+\s+S\.C\.R\.", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def create_chunks(pages):
    chunks = []

    words = []

    for page in pages:
        page_number = page["page"]
        text = clean_text(page["text"])

        page_words = text.split()

        for word in page_words:
            words.append((word, page_number))

    start = 0
    chunk_number = 1

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))

        chunk_words = words[start:end]

        text = " ".join(word for word, _ in chunk_words)

        start_page = chunk_words[0][1]
        end_page = chunk_words[-1][1]

        chunks.append({
            "chunk_id": chunk_number,
            "start_page": start_page,
            "end_page": end_page,
            "word_count": len(chunk_words),
            "text": text
        })

        if end == len(words):
            break

        start = end - OVERLAP
        chunk_number += 1

    return chunks


files = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(".json")
]

print(f"Found {len(files)} documents.\n")

total_chunks = 0

for i, filename in enumerate(files, start=1):

    input_path = os.path.join(INPUT_DIR, filename)

    with open(input_path, "r", encoding="utf-8") as f:
        document = json.load(f)

    chunks = create_chunks(document["pages"])

    output = {
        "document_id": document["document_id"],
        "filename": document["filename"],
        "page_count": document["page_count"],
        "chunk_count": len(chunks),
        "chunks": chunks
    }

    output_filename = filename

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    total_chunks += len(chunks)

    print(
        f"[{i}/{len(files)}] "
        f"{document['document_id']} → {len(chunks)} chunks"
    )


print("\nChunking complete.")
print(f"Documents processed: {len(files)}")
print(f"Total chunks created: {total_chunks}")
print(f"Chunks saved to: {OUTPUT_DIR}")