# Legal Document Retrieval Using NLP and FAISS

An NLP-powered legal document retrieval system that identifies semantically similar Indian Supreme Court judgments based on an uploaded legal PDF. The system uses Sentence Transformers to generate semantic embeddings and FAISS to perform similarity search.

## 1. Project Overview

Legal documents often contain complex terminology and lengthy discussions. Traditional keyword-based search may fail to identify relevant judgments when similar legal concepts are expressed using different words.

This project addresses this limitation through semantic similarity-based retrieval. Users can upload a legal PDF and retrieve potentially relevant judgments from an existing collection of Supreme Court judgments.

## 2. Key Features

- Upload legal documents in PDF format.
- Extract text from multi-page PDF documents.
- Divide legal documents into overlapping text chunks.
- Generate semantic embeddings using Sentence Transformers.
- Perform vector similarity search using FAISS.
- Retrieve and rank similar judgments.
- Display similarity scores and matching text passages.
- Present results through a Streamlit web interface.
- Support legal category detection if the category detection module is enabled.

## 3. Motivation

To simplify preliminary legal research by identifying relevant judgments based on the semantic meaning of legal documents rather than relying entirely on exact keyword matching.

## 4. Existing Gap

Traditional keyword-based retrieval systems have difficulty identifying semantically related documents when different terminology is used to express similar legal concepts.

This project uses NLP-based semantic embeddings and vector similarity search to improve the discovery of potentially relevant judgments.

## 5. System Architecture

```text
          Legal PDF Dataset
                  |
                  v
          PDF Text Extraction
                  |
                  v
             Text Chunking
          (500 words, 100 overlap)
                  |
                  v
        Sentence Transformer Model
            all-MiniLM-L6-v2
                  |
                  v
          384-Dimensional Embeddings
                  |
                  v
              FAISS Index
                  |
                  |
          User Uploads Legal PDF
                  |
                  v
          PDF Text Extraction
                  |
                  v
             Text Chunking
                  |
                  v
          Embedding Generation
                  |
                  v
          FAISS Similarity Search
                  |
                  v
         Similarity-Based Ranking
                  |
                  v
          Retrieved Judgments
                  |
                  v
            Streamlit Interface
```

## 6. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| Sentence Transformers | Semantic embedding generation |
| all-MiniLM-L6-v2 | Pre-trained embedding model |
| FAISS | Vector indexing and similarity search |
| PyMuPDF | PDF text extraction |
| NumPy | Numerical operations and embedding storage |
| JSON | Chunk metadata storage |

## 7. Dataset

The current implementation uses 200 English Supreme Court judgments from 2023.

| Parameter | Value |
|---|---|
| Court | Supreme Court of India |
| Year | 2023 |
| Number of judgments | 200 |
| Total text chunks | 4,157 |
| Chunk size | 500 words |
| Chunk overlap | 100 words |
| Embedding model | all-MiniLM-L6-v2 |
| Embedding dimension | 384 |
| FAISS vectors | 4,157 |

### Generated Files

| File | Description |
|---|---|
| `data/embeddings/embeddings.npy` | Stores the generated embedding vectors |
| `data/embeddings/chunks_metadata.json` | Stores chunk and judgment metadata |
| `data/index/legal_judgments.index` | Stores the FAISS vector index |

## 8. NLP Pipeline

1. **PDF Text Extraction:** Extracts text from legal judgments using PyMuPDF.
2. **Text Chunking:** Divides documents into 500-word chunks with a 100-word overlap.
3. **Embedding Generation:** Converts each chunk into a 384-dimensional vector using `all-MiniLM-L6-v2`.
4. **Vector Indexing:** Stores the embeddings in a FAISS index.
5. **Query Processing:** Extracts and processes text from the uploaded legal PDF.
6. **Similarity Search:** Compares query embeddings against stored judgment embeddings.
7. **Result Aggregation:** Groups matching chunks by their corresponding judgments.
8. **Result Display:** Presents ranked judgments, similarity scores, and relevant passages.

## 9. Similarity Search

The project uses FAISS `IndexFlatIP` for exact inner-product similarity search.

Since the embeddings are normalized, inner product is equivalent to cosine similarity.

\[
Similarity(A,B)=A\cdot B
\]

Higher similarity scores indicate greater semantic similarity between the retrieved text and the uploaded document.

## 10. Users, Inputs and Outputs

### Intended Users
- Law students
- Legal researchers
- Lawyers and legal professionals
- Academic researchers

### Input
- Legal documents in PDF format.
- Extracted text from uploaded documents.
- Numerical embeddings generated from document chunks.

### Output
- Top 5 potentially similar judgments.
- Similarity scores.
- Matching legal text passages.
- Available judgment metadata.
- Detected legal categories, if enabled.

## 11. Installation and Execution

### Prerequisites

- Python 3.10 or later
- pip
- Git

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd legal-document-retrieval
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies



```bash
pip install -r requirements.txt
```

### Step 4: Verify Project Data

Ensure the following files are available:

```text
data/
├── embeddings/
│   ├── embeddings.npy
│   └── chunks_metadata.json
└── index/
    └── legal_judgments.index
```

These files are required for retrieval using the existing indexed dataset.

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 12. Project Structure

```text
legal-document-retrieval/
│
├── app.py
│
├── scripts/
│   ├── search_cases.py
│   └── category_detection.py
│
├── data/
│   ├── embeddings/
│   │   ├── embeddings.npy
│   │   └── chunks_metadata.json
│   │
│   └── index/
│       └── legal_judgments.index
│
├── requirements.txt
├── README.md
└── .gitignore
```

*The structure above represents the main project components; adjust filenames and folders to match the actual repository.*

## 13. Limitations

- The current reference dataset contains 200 Supreme Court judgments from 2023.
- Retrieval quality depends on the available dataset and embedding model.
- Similarity scores indicate semantic similarity, not legal applicability.
- The system does not replace professional legal advice or independent examination of original judgments.
- Scanned PDFs without an embedded text layer may require OCR processing.

## 14. Future Enhancements

- Expand the dataset to include High Court and other Indian court judgments.
- Integrate OCR for scanned legal documents.
- Improve legal-domain-specific embedding models.
- Add advanced filtering by court, year, and legal category.
- Introduce improved document-level relevance evaluation.
- Support multilingual legal document retrieval.

## 15. References

1. Reimers, N., and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. https://doi.org/10.18653/v1/D19-1410
2. Johnson, J., Douze, M., and Jégou, H. (2017). Billion-scale Similarity Search with GPUs. https://arxiv.org/abs/1702.08734
3. Sentence Transformers: https://sbert.net/
4. FAISS: https://github.com/facebookresearch/faiss
5. PyMuPDF: https://pymupdf.readthedocs.io/
6. Streamlit: https://docs.streamlit.io/

---

**Note:** This project is an NLP-based semantic legal document retrieval prototype developed for academic and research purposes.
