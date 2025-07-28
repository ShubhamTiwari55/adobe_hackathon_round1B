
# Approach Explanation

## 1. Problem Framing  
Round 1B asks the system to return the *k* most relevant sections—and their high-value subsections—from a small collection of PDFs, conditioned on a persona and a job-to-be-done. Relevance therefore depends on:

1. Topical similarity between the query (persona + task) and the page text  
2. Document structure (headings often summarize intent)  
3. Internal ranking among the selected pages and subsections

## 2. Pipeline Overview

| Stage              | Component                           | Purpose |
|--------------------|-------------------------------------|---------|
| Pre-processing     | `extract_pages()`                   | Extracts raw text with PyMuPDF; falls back to OCR via Tesseract if needed and also segments candidate *subsections* using regex heuristics for numbered/bold headings |
| Sparse Search      | BM25 (`rank_bm25`)                  | Fast lexical matching to create BM25 scores and an index used later for subsection scoring |
| Dense Search       | Sentence-Transformers (`all-MiniLM-L6-v2`) | Generates 384-dim embeddings for all pages and the query; cosine similarity provides a semantic feature |
| Hand-crafted       | `HashingVectorizer TF` and `is_heading` flag | Adds exact-term frequency and a structural prior |
| Learning to Rank   | LightGBM classifier                 | Trains on labelled collections to predict page relevance. Final probability is thresholded (F1-optimized) and stored in `lgbm_relevance.joblib` |
| Cross-Encoder Re-rank | `ms-marco-MiniLM-L-6-v2`         | Jointly encodes (query, page_text) pairs to refine the top candidates with a deeper interaction model |
| Sub-section Analysis | Same Cross-Encoder + BM25         | Reranks detected subsections on the shortlisted pages and outputs an ordered list with auxiliary metrics |

## 3. Feature Set  
For every page, we build a 4-dimensional feature vector:

```
Xᵢ = [cosine, bm25, tf, is_head]
```

The blend of sparse, dense, and structural signals proved robust across heterogeneous document styles (guides, reports, textbooks) while adding negligible latency.

## 4. Model Training & Thresholding  
Three labelled collections were parsed; class imbalance (~5%) was mitigated with SMOTE oversampling before an 80/20 split and LightGBM training. A sweep over 18 candidate thresholds maximized F1 on the validation fold. Both the classifier and optimal threshold are persisted. The resulting offline artifact is 15 MB—well below the 1 GB limit.

## 5. Runtime Flow  
At inference, the container:

1. Loads the two HF models and the joblib bundle from `app/model/`
2. Streams each PDF page, emitting text and structural hints on the fly (no full memory load)
3. Computes base probabilities, keeps rows where *p ≥ τ/5*, and passes them through the Cross-Encoder (batch size = 16 for CPU efficiency)
4. Emits the *top-k* pages as `extracted_sections` and ranks their internal subsections

> **End-to-end time on a 5-PDF (≈150 pages) collection is ~38s on an 8-core AMD64 CPU.**  
> The overall model footprint is <130 MB—meeting every constraint.

## 6. Strengths & Future Work  
- ✅ Multi-stage retrieval balances speed (BM25) and depth (Cross-Encoder) without GPU reliance  
- ✅ Regex-based subsection carving is language-agnostic and fast  
- 🔄 Could be improved with a layout-aware classifier in Round 2  
- 🌐 Extending to low-resource languages would only require a multilingual encoder swap  

---

This architecture delivers **high recall**, **precise ranking**, and **deterministic performance** within the hackathon limits.
