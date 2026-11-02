
import os
import sys
import re
import json

from collections import Counter, defaultdict

import streamlit as st
import fitz
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS_DIR = os.path.join(
    BASE_DIR,
    "scripts"
)

INDEX_PATH = os.path.join(
    BASE_DIR,
    "data",
    "index",
    "legal_judgments.index"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "embeddings",
    "chunks_metadata.json"
)

MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

CANDIDATE_K = 30
TOP_CASES = 5
TOP_CHUNKS_PER_CASE = 3


# ==================================================
# CATEGORY DETECTION IMPORT
# ==================================================

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

try:

    from category_detection import detect_categories

except ImportError:

    def detect_categories(text):
        return {}


# ==================================================
# STREAMLIT CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Indian Legal Case Analysis",
    page_icon="⚖️",
    layout="wide"
)


# ==================================================
# LOAD MODEL, FAISS INDEX AND METADATA
# ==================================================

@st.cache_resource
def load_resources():

    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Loading FAISS index...")

    index = faiss.read_index(
        INDEX_PATH
    )

    print("Loading chunk metadata...")

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    if index.ntotal != len(metadata):

        raise ValueError(
            f"FAISS contains {index.ntotal} vectors, "
            f"but metadata contains {len(metadata)} chunks."
        )

    return model, index, metadata


# ==================================================
# PDF TEXT EXTRACTION
# ==================================================

def extract_pdf_text(uploaded_file):

    """
    Extract text from an uploaded PDF.

    Returns:
        List containing page number and extracted text.
    """

    pages = []

    document = fitz.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    )

    try:

        for page_number, page in enumerate(
            document,
            start=1
        ):

            text = page.get_text(
                "text"
            ).strip()

            if text:

                pages.append({

                    "page": page_number,

                    "text": text

                })

    finally:

        document.close()

    return pages


# ==================================================
# DOCUMENT CHUNKING
# ==================================================

def create_upload_chunks(pages):

    """
    Create overlapping word-based chunks.

    Chunk size:
        500 words

    Overlap:
        100 words

    Page metadata is preserved.
    """

    words_with_pages = []

    for page in pages:

        words = page["text"].split()

        for word in words:

            words_with_pages.append(
                (
                    word,
                    page["page"]
                )
            )

    chunks = []

    step = CHUNK_SIZE - CHUNK_OVERLAP

    for start in range(
        0,
        len(words_with_pages),
        step
    ):

        items = words_with_pages[
            start:start + CHUNK_SIZE
        ]

        if not items:

            break

        chunk_text = " ".join(
            word for word, page in items
        )

        chunks.append({

            "text": chunk_text,

            "start_page": items[0][1],

            "end_page": items[-1][1],

            "word_count": len(items)

        })

        if (
            start + CHUNK_SIZE
            >= len(words_with_pages)
        ):

            break

    return chunks


# ==================================================
# DOCUMENT SUMMARY
# ==================================================

STOPWORDS = {

    "the", "a", "an", "and", "or", "but",

    "if", "is", "are", "was", "were", "be",

    "been", "of", "to", "in", "on", "for",

    "with", "by", "from", "at", "as", "that",

    "this", "it", "he", "she", "they", "them",

    "his", "her", "their", "its", "we", "our",

    "you", "your", "which", "who", "whom",

    "what", "when", "where", "while", "after",

    "before", "into", "has", "have", "had",

    "not", "can", "could", "would", "should",

    "shall", "will", "may", "might", "also",

    "such", "than", "then", "there", "here",

    "all", "any", "some", "being", "do",

    "does", "did", "so", "up", "out", "about",

    "over", "under", "between", "court",

    "honble", "judgment", "judgement",

    "petitioner", "respondent"

}


def summarize_document(
    text,
    num_sentences=8
):

    """
    Generate an extractive summary.

    Important sentences are selected from the
    uploaded document using word frequency.

    No external API is used.
    """

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if not text:

        return "No text available for summarization."

    # Sentence splitting
    sentences = re.split(
        r'(?<=[.!?])\s+(?=[A-Z0-9"\'])',
        text
    )

    sentences = [

        sentence.strip()

        for sentence in sentences

        if len(
            sentence.strip().split()
        ) >= 8

    ]

    if not sentences:

        return text[:1500]

    # Remove duplicate sentences
    unique_sentences = []

    seen = set()

    for sentence in sentences:

        normalized = sentence.lower()

        if normalized not in seen:

            seen.add(normalized)

            unique_sentences.append(
                sentence
            )

    sentences = unique_sentences

    # Calculate word frequency
    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text.lower()
    )

    word_frequency = Counter(

        word

        for word in words

        if word not in STOPWORDS
        and len(word) > 2

    )

    if not word_frequency:

        return " ".join(
            sentences[:num_sentences]
        )

    max_frequency = max(
        word_frequency.values()
    )

    # Normalize frequency
    for word in word_frequency:

        word_frequency[word] /= max_frequency

    sentence_scores = []

    # Score sentences
    for position, sentence in enumerate(
        sentences
    ):

        sentence_words = re.findall(
            r"\b[a-zA-Z]+\b",
            sentence.lower()
        )

        score = sum(

            word_frequency.get(
                word,
                0
            )

            for word in sentence_words

        )

        # Penalize long sentences
        if len(sentence_words) > 60:

            score /= len(sentence_words)

        # Slight preference for beginning
        if position < 5:

            score *= 1.15

        sentence_scores.append(

            (
                position,
                score,
                sentence
            )

        )

    # Select important sentences
    ranked_sentences = sorted(

        sentence_scores,

        key=lambda item: item[1],

        reverse=True

    )

    selected = ranked_sentences[
        :num_sentences
    ]

    # Restore document order
    selected.sort(
        key=lambda item: item[0]
    )

    summary = " ".join(

        sentence

        for position, score, sentence
        in selected

    )

    return summary


# ==================================================
# LEGAL INFORMATION EXTRACTION
# ==================================================

def extract_legal_information(text):

    """
    Extract basic legal information using regex.

    This is rule-based extraction and may produce
    incomplete or incorrect matches.
    """

    information = {}

    # ----------------------------------------------
    # CASE NUMBERS
    # ----------------------------------------------

    case_number_patterns = [

        r"(?:case|appeal|petition|writ)"
        r"\s*(?:no\.?|number)"
        r"\s*[:\-]?\s*"
        r"([A-Z0-9/\-()]+)",

        r"\b(?:SLP|Crl\.?A\.?|C\.?A\.?|"
        r"W\.?P\.?|CRL\.?M\.?C\.?|FAO)"
        r"\s*(?:No\.?)?\s*"
        r"([A-Z0-9/\-()]+)"

    ]

    case_numbers = []

    for pattern in case_number_patterns:

        matches = re.findall(

            pattern,

            text,

            flags=re.IGNORECASE

        )

        case_numbers.extend(
            matches
        )

    information["Case Numbers"] = list(

        dict.fromkeys(
            case_numbers
        )

    )[:10]

    # ----------------------------------------------
    # ACTS
    # ----------------------------------------------

    acts = re.findall(

        r"\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,8}"
        r"\s+Act(?:,\s*\d{4})?\b",

        text

    )

    information["Acts"] = list(

        dict.fromkeys(
            acts
        )

    )[:15]

    # ----------------------------------------------
    # LEGAL SECTIONS
    # ----------------------------------------------

    sections = re.findall(

        r"\bSections?\s+\d+[A-Za-z]*"
        r"(?:\s*\([^)]+\))?",

        text,

        flags=re.IGNORECASE

    )

    information["Sections"] = list(

        dict.fromkeys(
            sections
        )

    )[:20]

    # ----------------------------------------------
    # COURTS
    # ----------------------------------------------

    court_patterns = [

        r"Supreme Court of India",

        r"High Court of [A-Za-z ]+",

        r"District Court",

        r"Sessions Court",

        r"Family Court"

    ]

    courts = []

    for pattern in court_patterns:

        matches = re.findall(

            pattern,

            text,

            flags=re.IGNORECASE

        )

        courts.extend(
            matches
        )

    information["Courts"] = list(

        dict.fromkeys(
            courts
        )

    )[:10]

    return information


# ==================================================
# CATEGORY DETECTION
# ==================================================

def get_categories(text):

    """
    Detect legal categories using the
    existing category_detection.py module.
    """

    try:

        return detect_categories(
            text
        )

    except Exception:

        return {}


# ==================================================
# FAISS SIMILARITY SEARCH
# ==================================================

def find_similar_cases(
    query_chunks,
    model,
    index,
    metadata
):

    """
    Search uploaded document chunks against
    the existing FAISS judgment index.

    Retrieved chunks are grouped by judgment.
    """

    grouped = defaultdict(list)

    for query_chunk in query_chunks:

        query_embedding = model.encode(

            [query_chunk["text"]],

            normalize_embeddings=True,

            convert_to_numpy=True

        )

        query_embedding = np.asarray(

            query_embedding,

            dtype=np.float32

        )

        k = min(

            CANDIDATE_K,

            index.ntotal

        )

        scores, indices = index.search(

            query_embedding,

            k

        )

        for score, idx in zip(

            scores[0],

            indices[0]

        ):

            if idx < 0 or idx >= len(metadata):

                continue

            chunk = metadata[idx]

            document_id = chunk.get(

                "document_id",

                chunk.get(
                    "filename",
                    f"unknown_{idx}"
                )

            )

            grouped[document_id].append({

                "score": float(score),

                "chunk": chunk

            })

    results = []

    # ----------------------------------------------
    # GROUP AND RANK JUDGMENTS
    # ----------------------------------------------

    for document_id, matches in grouped.items():

        unique_chunks = {}

        for match in matches:

            chunk = match["chunk"]

            chunk_id = chunk.get(
                "chunk_id"
            )

            if chunk_id is not None:

                key = chunk_id

            else:

                key = (

                    chunk.get("filename"),

                    chunk.get("start_page"),

                    chunk.get("end_page"),

                    chunk.get("text", "")[:100]

                )

            if (

                key not in unique_chunks

                or match["score"]
                > unique_chunks[key]["score"]

            ):

                unique_chunks[key] = match

        ranked_matches = sorted(

            unique_chunks.values(),

            key=lambda item: item["score"],

            reverse=True

        )

        best_matches = ranked_matches[
            :TOP_CHUNKS_PER_CASE
        ]

        if not best_matches:

            continue

        case_score = float(

            np.mean([

                item["score"]

                for item in best_matches

            ])

        )

        # Category detection from retrieved passages
        combined_text = " ".join(

            item["chunk"].get(
                "text",
                ""
            )

            for item in best_matches

        )

        categories = get_categories(
            combined_text
        )

        results.append({

            "document_id": document_id,

            "score": case_score,

            "matches": best_matches,

            "categories": categories

        })

    results.sort(

        key=lambda item: item["score"],

        reverse=True

    )

    return results[:TOP_CASES]


# ==================================================
# DISPLAY CATEGORY INFORMATION
# ==================================================


# ==================================================
# DISPLAY DETECTED LEGAL CATEGORIES
# ==================================================


def display_categories(categories):

    st.markdown("""
    <style>
    .category-box {
        background-color: #172B40;
        border: 2px solid #3B82F6;
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 20px;
    }

    .category-item {
        background-color: #243B53;
        border-radius: 8px;
        padding: 15px;
        margin-top: 12px;
    }

    .category-title {
        color: #60A5FA;
        font-size: 20px;
        font-weight: bold;
    }

    .category-keywords {
        color: #D1D5DB;
        margin-top: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="category-box">'
        '<h2 style="color:white;">'
        'Detected Legal Categories'
        '</h2>',
        unsafe_allow_html=True
    )

    if not categories:
        st.info("No legal categories detected.")
        return

    for category, keywords in categories.items():

        st.markdown(
            f"""
            <div class="category-item">
                <div class="category-title">
                    {category}
                </div>
                <div class="category-keywords">
                    <b>Matching keywords:</b>
                    {", ".join(keywords)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)




# ==================================================
# DISPLAY LEGAL INFORMATION
# ==================================================

def display_legal_information(
    legal_info
):

    st.subheader(
        "Legal Information"
    )

    # Case numbers
    if legal_info["Case Numbers"]:

        st.write(
            "**Case Number(s):**"
        )

        for number in legal_info["Case Numbers"]:

            st.write(
                f"- {number}"
            )

    # Courts
    if legal_info["Courts"]:

        st.write(
            "**Court(s) Mentioned:**"
        )

        for court in legal_info["Courts"]:

            st.write(
                f"- {court}"
            )

    # Acts
    if legal_info["Acts"]:

        st.write(
            "**Acts Mentioned:**"
        )

        for act in legal_info["Acts"]:

            st.write(
                f"- {act}"
            )

    # Sections
    if legal_info["Sections"]:

        st.write(
            "**Legal Sections Mentioned:**"
        )

        for section in legal_info["Sections"]:

            st.write(
                f"- {section}"
            )

    if not any(

        legal_info[key]

        for key in legal_info

    ):

        st.write(
            "No legal information was detected."
        )


# ==================================================
# DISPLAY SIMILAR JUDGMENTS
# ==================================================

def display_results(results):

    st.subheader(
        "Similar Judgments"
    )

    st.caption(

        "Ranking scores measure aggregated vector similarity, "
        "not the probability of legal relevance."

    )

    if not results:

        st.warning(
            "No similar judgments found."
        )

        return

    for rank, result in enumerate(

        results,

        start=1

    ):

        # Use normal container for compatibility
        with st.container():

            st.markdown(
                f"### {rank}. {result['document_id']}"
            )

            st.write(

                "**Similarity ranking score:** "

                f"{result['score']:.4f}"

            )

            first_chunk = result["matches"][0]["chunk"]

            filename = first_chunk.get(

                "filename",

                "Unknown"

            )

            st.write(
                f"**Source file:** {filename}"
            )

            # --------------------------------------
            # CATEGORY
            # --------------------------------------

            st.write(
                "**Detected categories:**"
            )

            if result["categories"]:

                display_categories(
                    result["categories"]
                )

            else:

                st.write(
                    "No categories detected."
                )

            # --------------------------------------
            # PASSAGES
            # --------------------------------------

            st.write(
                "**Relevant passages:**"
            )

            for match in result["matches"]:

                chunk = match["chunk"]

                start_page = chunk.get(
                    "start_page",
                    "?"
                )

                end_page = chunk.get(
                    "end_page",
                    "?"
                )

                with st.expander(

                    f"Pages {start_page}-{end_page} "
                    f"| Similarity: {match['score']:.4f}"

                ):

                    st.write(

                        chunk.get(
                            "text",
                            ""
                        )

                    )

            st.divider()


# ==================================================
# MAIN APPLICATION
# ==================================================

def main():

    # ----------------------------------------------
    # HEADER
    # ----------------------------------------------

    st.title(
        "Indian Legal Case Analysis"
    )

    st.write(

        "Upload a legal document to analyze its contents "
        "and find similar Indian court judgments."

    )



    # ----------------------------------------------
    # FILE UPLOAD
    # ----------------------------------------------

    uploaded_file = st.file_uploader(

        "Upload a legal document (PDF)",

        type=["pdf"]

    )

    if uploaded_file is None:

        st.info(
            "Upload a PDF to begin document analysis."
        )

        return

    st.write(
        f"**Uploaded file:** {uploaded_file.name}"
    )

    # ----------------------------------------------
    # ANALYSIS BUTTON
    # ----------------------------------------------

    if not st.button(

        "Analyze Document and Find Similar Cases",

        type="primary"

    ):

        return

    try:

        # ------------------------------------------
        # LOAD RESOURCES
        # ------------------------------------------

        with st.spinner(

            "Loading embedding model and FAISS index..."

        ):

            model, index, metadata = load_resources()

        # ------------------------------------------
        # EXTRACT TEXT
        # ------------------------------------------

        with st.spinner(

            "Extracting text from uploaded PDF..."

        ):

            pages = extract_pdf_text(
                uploaded_file
            )

        if not pages:

            st.error(

                "No readable text was extracted. "
                "The document may be scanned. "
                "OCR is not enabled in this version."

            )

            return

        # ------------------------------------------
        # DOCUMENT TEXT
        # ------------------------------------------

        full_text = " ".join(

            page["text"]

            for page in pages

        )

        word_count = len(
            full_text.split()
        )

        # ------------------------------------------
        # DOCUMENT CHUNKS
        # ------------------------------------------

        with st.spinner(

            "Preparing document chunks..."

        ):

            query_chunks = create_upload_chunks(
                pages
            )

        if not query_chunks:

            st.error(
                "No text chunks could be created."
            )

            return

        # ------------------------------------------
        # DOCUMENT SUMMARY
        # ------------------------------------------

        with st.spinner(

            "Summarizing uploaded document..."

        ):

            summary = summarize_document(

                full_text,

                num_sentences=8

            )

        # ------------------------------------------
        # LEGAL INFORMATION
        # ------------------------------------------

        legal_info = extract_legal_information(
            full_text
        )

        # ------------------------------------------
        # DOCUMENT CATEGORIES
        # ------------------------------------------

        document_categories = get_categories(
            full_text
        )

        # ------------------------------------------
        # DOCUMENT ANALYSIS
        # ------------------------------------------

        st.success(
            "Document text extracted successfully."
        )

        st.subheader(
            "Uploaded Document Analysis"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Pages with text",
            len(pages)
        )

        col2.metric(
            "Word count",
            f"{word_count:,}"
        )

        col3.metric(
            "Query chunks",
            len(query_chunks)
        )

        # ------------------------------------------
        # DETECTED LEGAL CATEGORIES (FIRST)
        # ------------------------------------------

        display_categories(
            document_categories
        )

        # ------------------------------------------
        # SUMMARY
        # ------------------------------------------

        st.subheader(
            "Document Summary"
        )

        st.write(
            summary
        )

        st.caption(

            "Extractive summary: important sentences "
            "selected from the uploaded document. "
            "Verify against the original PDF."

        )

        # ------------------------------------------
        # LEGAL INFORMATION
        # ------------------------------------------

        display_legal_information(
            legal_info
        )

        # ------------------------------------------
        # SIMILAR CASE SEARCH
        # ------------------------------------------

        st.subheader(
            "Searching Similar Judgments"
        )

        with st.spinner(

            "Comparing uploaded document with indexed judgments..."

        ):

            results = find_similar_cases(

                query_chunks,

                model,

                index,

                metadata

            )

        # ------------------------------------------
        # DISPLAY RESULTS
        # ------------------------------------------

        display_results(
            results
        )

        # ------------------------------------------
        # FOOTER
        # ------------------------------------------

        st.caption(

            "Research assistance only. Similarity scores "
            "do not establish legal relevance or predict "
            "case outcomes. Verify judgments and citations "
            "against their original sources."

        )

    except Exception as error:

        st.error(
            f"An error occurred: {error}"
        )


# ==================================================
# APPLICATION ENTRY POINT
# ==================================================

if __name__ == "__main__":

    main()


