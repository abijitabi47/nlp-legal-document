import os
import requests
import xml.etree.ElementTree as ET

BUCKET_URL = "https://indian-supreme-court-judgments.s3.amazonaws.com"
PREFIX = "data/pdf/year=2023/english/"
OUTPUT_DIR = "data/raw"
LIMIT = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Finding judgments...")

params = {
    "list-type": "2",
    "prefix": PREFIX,
    "max-keys": LIMIT
}

response = requests.get(BUCKET_URL, params=params)
response.raise_for_status()

root = ET.fromstring(response.content)

namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

keys = []

for content in root.findall("s3:Contents", namespace):
    key = content.find("s3:Key", namespace).text
    if key.lower().endswith(".pdf"):
        keys.append(key)

print(f"Found {len(keys)} judgments.")

for i, key in enumerate(keys, start=1):
    filename = os.path.basename(key)
    output_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(output_path):
        print(f"[{i}/{len(keys)}] Already exists: {filename}")
        continue

    url = f"{BUCKET_URL}/{key}"

    print(f"[{i}/{len(keys)}] Downloading {filename}")

    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

print("\nDownload complete.")
print(f"Files saved to: {OUTPUT_DIR}")