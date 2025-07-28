import pathlib
import json
import time
import re
import os
import numpy as np
import tqdm
import joblib

# PDF and Image Processing
import pymupdf  # Replaces fitz
import cv2
import pytesseract
from PIL import Image

# Machine Learning and Text Processing
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import HashingVectorizer

# --- Model and Feature Extractor Initialization ---
# These models are loaded once and reused.
# IMPORTANT: The paths are updated to load from a local 'models' directory.
print("Loading models...")
try:
    EMB = SentenceTransformer('./models/all-MiniLM-L6-v2')
    CE = CrossEncoder('./models/ms-marco-MiniLM-L-6-v2')
    HV = HashingVectorizer(stop_words='english', n_features=2**20)
    MODEL_PACK = joblib.load('lgbm_relevance.joblib')
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load models. Ensure 'lgbm_relevance.joblib' and the './models' directory exist. Error: {e}")
    exit() # Exit if models can't be loaded

# --- Regular Expressions for Text Structuring ---
HEAD = re.compile(r'^([0-9]+(\\.[0-9]+)*|[IVXLCDM]+)\\s+|^[A-Z][A-Z \\-]{4,}')
SUBSECTION_HEAD = re.compile(r'^(\\s*[0-9]+\\.\\s+.*|^[A-Z]\\.\\s+.*|^\\*\\*\\s*.*\\s*\\*\\*\\s*)')


# --- Core Functions for PDF Processing and Feature Engineering ---

def page_text(pg):
    """Extracts text from a PDF page, using OCR as a fallback."""
    txt = pg.get_text('text', sort=True).strip()
    if txt:
        return txt

    # Fallback to OCR if no text is found
    try:
        pix = pg.get_pixmap()
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.alpha:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
        return pytesseract.image_to_string(Image.fromarray(img_data), lang='eng')
    except Exception:
        return "" # Return empty string if OCR also fails

def extract_pages(pdf_path):
    """Yields processed page data, including text and identified subsections."""
    doc = pymupdf.open(pdf_path)
    for pg in doc:
        text = page_text(pg)
        if not text:
            continue

        lines = text.splitlines()
        subsections = []
        current_subsection_start = 0
        for i, line in enumerate(lines):
            if SUBSECTION_HEAD.match(line) and i > 0:
                subsection_text = "\\n".join(lines[current_subsection_start:i]).strip()
                if subsection_text:
                    subsections.append({"text": subsection_text, "start_line": current_subsection_start + 1})
                current_subsection_start = i

        last_subsection_text = "\\n".join(lines[current_subsection_start:]).strip()
        if last_subsection_text:
            subsections.append({"text": last_subsection_text, "start_line": current_subsection_start + 1})

        yield {
            "doc": pdf_path.stem,
            "page": pg.number + 1,
            "text": text,
            "is_head": bool(HEAD.match(lines[0] if lines else '')),
            "subsections": subsections
        }

def build_index(texts):
    """Builds a BM25 index from a list of texts."""
    return BM25Okapi([t.split() for t in texts])

def features(pages, query, bm25):
    """Generates features for the LightGBM model."""
    texts = [p["text"] for p in pages]
    q_emb = EMB.encode([query])[0]
    p_emb = EMB.encode(texts, convert_to_numpy=True)

    cos_sim = (p_emb @ q_emb)
    bm25_scores = np.array(bm25.get_scores(query.split()))
    tf_scores = HV.transform(texts).dot(HV.transform([query]).T).toarray().ravel()
    struct_scores = np.array([p["is_head"] for p in pages], dtype=float)

    return np.vstack([cos_sim, bm25_scores, tf_scores, struct_scores]).T


# --- Main Application Logic ---

def rank_collection(collection_path, output_dir, top_k=10):
    """
    Ranks documents in a single collection based on persona and job, and writes the output JSON.
    """
    meta = json.load(open(collection_path / 'challenge1b_input.json'))
    query = f'{meta["persona"]["role"]} {meta["job_to_be_done"]["task"]}'

    pages = []
    pdf_dir = collection_path / 'PDFs'
    for pdf in pdf_dir.glob('*.pdf'):
        pages.extend(list(extract_pages(pdf)))

    if not pages:
        print(f"⚠️ No pages extracted from PDFs in {collection_path.name}. Skipping.")
        return

    # Stage 1: Candidate Retrieval
    bm25 = build_index([p["text"] for p in pages])
    X = features(pages, query, bm25)
    base_scores = MODEL_PACK['clf'].predict_proba(X)[:, 1]

    # Select candidates for re-ranking with a lower threshold
    candidate_indices = np.where(base_scores >= MODEL_PACK['thr'] / 5)[0]

    # Stage 2: Cross-Encoder Re-ranking
    ce_scores = CE.predict([[query, pages[i]["text"][:4096]] for i in candidate_indices], batch_size=32)
    reranked_pages = sorted(zip(candidate_indices, ce_scores), key=lambda x: x[1], reverse=True)[:top_k]

    # --- Generate Final Output ---
    output_sections = []
    for rank, (page_idx, score) in enumerate(reranked_pages, 1):
        p = pages[page_idx]
        output_sections.append({
            "document": p["doc"] + '.pdf',
            "page_number": p["page"],
            "section_title": p["text"].splitlines()[0][:120],
            "importance_rank": rank
        })

    # Subsection Analysis
    subsection_analysis_list = []
    for page_idx, _ in reranked_pages:
        p = pages[page_idx]
        subsections = p.get("subsections", [])
        if not subsections:
            continue

        sub_texts = [sub["text"][:4096] for sub in subsections if sub["text"].strip()]
        if not sub_texts:
            continue

        # Score subsections for relevance
        ce_scores_sub = CE.predict([[query, sub_text] for sub_text in sub_texts], batch_size=32)

        for i, sub in enumerate(subsections):
            if sub["text"].strip():
                subsection_analysis_list.append({
                    "Document": p["doc"] + '.pdf',
                    "Page Number": p["page"],
                    "subsection_start_line": sub["start_line"],
                    "relevance_score_ce": float(ce_scores_sub[i]),
                    "Refined Text": sub["text"]
                })

    # Sort subsections by relevance score
    subsection_analysis_list = sorted(subsection_analysis_list, key=lambda x: x["relevance_score_ce"], reverse=True)

    # Prepare final JSON content
    final_output = {
        "metadata": {
            "input_documents": [f.name for f in pdf_dir.glob('*.pdf')],
            "persona": meta["persona"],
            "job_to_be_done": meta["job_to_be_done"],
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        },
        "extracted_sections": output_sections,
        "subsection_analysis": subsection_analysis_list
    }

    # Write output to file
    output_file_path = output_dir / f"{collection_path.name}_output.json"
    with open(output_file_path, 'w') as f:
        json.dump(final_output, f, indent=2)
    print(f'🎉 Output written for {collection_path.name} to {output_file_path}')

def process_all_collections(input_dir_str, output_dir_str):
    """Processes all collections from the input directory and saves results to the output directory."""
    input_path = pathlib.Path(input_dir_str)
    output_path = pathlib.Path(output_dir_str)
    output_path.mkdir(exist_ok=True)

    if not any(input_path.iterdir()):
        print("Input directory is empty. Nothing to process.")
        return

    for collection_path in input_path.iterdir():
        if collection_path.is_dir():
            print(f"\n--- Processing Collection: {collection_path.name} ---")
            try:
                rank_collection(collection_path, output_path, top_k=10)
            except Exception as e:
                print(f"❌ Error processing {collection_path.name}: {e}")

# --- Script Entry Point ---
if __name__ == '__main__':
    # These are the paths inside the Docker container
    INPUT_DIR = '/app/input'
    OUTPUT_DIR = '/app/output'

    print("Starting document intelligence processing...")
    process_all_collections(INPUT_DIR, OUTPUT_DIR)
    print("\n✅ All collections processed.")