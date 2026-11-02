import os
import fitz
import pandas as pd

DATA_DIR = "data/raw"

results = []

files = [
    f for f in os.listdir(DATA_DIR)
    if f.lower().endswith(".pdf")
]

print(f"Found {len(files)} PDF files.\n")

for filename in files:
    path = os.path.join(DATA_DIR, filename)

    try:
        doc = fitz.open(path)

        page_count = len(doc)
        text = ""

        for page in doc:
            text += page.get_text()

        text = text.strip()

        results.append({
            "filename": filename,
            "pages": page_count,
            "characters": len(text),
            "words": len(text.split()),
            "text_available": len(text) > 100
        })

        doc.close()

    except Exception as e:
        results.append({
            "filename": filename,
            "pages": 0,
            "characters": 0,
            "words": 0,
            "text_available": False
        })

df = pd.DataFrame(results)

print("Dataset validation complete.\n")

print(f"Total PDFs: {len(df)}")
print(f"Readable PDFs: {df['text_available'].sum()}")
print(f"Unreadable/empty PDFs: {(~df['text_available']).sum()}")
print(f"Average pages: {df['pages'].mean():.2f}")
print(f"Average words: {df['words'].mean():.2f}")

os.makedirs("data/metadata", exist_ok=True)

df.to_csv(
    "data/metadata/dataset_validation.csv",
    index=False
)

print("\nValidation report saved to:")
print("data/metadata/dataset_validation.csv")