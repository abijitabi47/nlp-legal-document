import os
import re
import json
import fitz

INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(".pdf")
]

print(f"Found {len(files)} PDFs\n")


def clean_text(text):
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    return text.strip()


for i, filename in enumerate(files, start=1):
    pdf_path = os.path.join(INPUT_DIR, filename)

    doc = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text()
        text = clean_text(text)

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    doc.close()

    document_id = os.path.splitext(filename)[0]

    output = {
        "document_id": document_id,
        "filename": filename,
        "page_count": len(pages),
        "pages": pages
    }

    output_path = os.path.join(
        OUTPUT_DIR,
        document_id + ".json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[{i}/{len(files)}] {filename}")


print("\nExtraction complete.")
print(f"Processed files saved to: {OUTPUT_DIR}")