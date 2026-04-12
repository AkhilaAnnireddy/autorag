import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
RESULTS_DIR = "results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "retrieval_results.csv")

# Use MPS on Apple Silicon
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Embedding model
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# 12 configurations
CHUNK_SIZES = [256, 512, 1024]
RETRIEVAL_METHODS = ["bm25", "dense", "hybrid", "rerank"]

CONFIGS = []
config_id = 1
for chunk_size in CHUNK_SIZES:
    for method in RETRIEVAL_METHODS:
        CONFIGS.append({
            "config_id": f"config_{config_id}",
            "chunk_size": chunk_size,
            "retrieval_method": method
        })
        config_id += 1

# ── Load models once ───────────────────────────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)

print("Loading reranker model...")
reranker = CrossEncoder(RERANK_MODEL, device=DEVICE)

# ── Chunking ───────────────────────────────────────────────────────────────────
def chunk_text(text, chunk_size):
    """Split text into chunks of approximately chunk_size tokens (words)."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

# ── Retrieval methods ──────────────────────────────────────────────────────────
def retrieve_bm25(chunks, question, top_k=1):
    tokenized_chunks = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = question.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

def retrieve_dense(chunks, question, top_k=1):
    chunk_embeddings = embedder.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    query_embedding = embedder.encode([question], show_progress_bar=False, convert_to_numpy=True)

    # Normalize for cosine similarity
    faiss.normalize_L2(chunk_embeddings)
    faiss.normalize_L2(query_embedding)

    dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(chunk_embeddings)

    _, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def retrieve_hybrid(chunks, question, top_k=1):
    """Reciprocal Rank Fusion of BM25 + Dense."""
    k = 60  # RRF constant

    # BM25 ranking
    tokenized_chunks = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    bm25_scores = bm25.get_scores(question.lower().split())
    bm25_ranking = np.argsort(bm25_scores)[::-1]

    # Dense ranking
    chunk_embeddings = embedder.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    query_embedding = embedder.encode([question], show_progress_bar=False, convert_to_numpy=True)
    faiss.normalize_L2(chunk_embeddings)
    faiss.normalize_L2(query_embedding)
    dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(chunk_embeddings)
    _, dense_indices = index.search(query_embedding, len(chunks))
    dense_ranking = dense_indices[0]

    # RRF scores
    rrf_scores = np.zeros(len(chunks))
    for rank, idx in enumerate(bm25_ranking):
        rrf_scores[idx] += 1 / (k + rank + 1)
    for rank, idx in enumerate(dense_ranking):
        if idx < len(chunks):
            rrf_scores[idx] += 1 / (k + rank + 1)

    top_indices = np.argsort(rrf_scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

def retrieve_rerank(chunks, question, top_k=1, initial_k=5):
    """Dense retrieval followed by cross-encoder reranking."""
    # First get top initial_k with dense
    initial_k = min(initial_k, len(chunks))
    candidate_chunks = retrieve_dense(chunks, question, top_k=initial_k)

    if not candidate_chunks:
        return []

    # Rerank with cross-encoder
    pairs = [[question, chunk] for chunk in candidate_chunks]
    scores = reranker.predict(pairs)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [candidate_chunks[i] for i in top_indices]

# ── Retrieval precision ────────────────────────────────────────────────────────
def compute_retrieval_precision(chunks, qa_pairs, retrieval_method):
    """
    For each QA pair, retrieve top-1 chunk and check if it contains the answer.
    Returns precision as fraction of correct retrievals.
    """
    correct = 0
    total = len(qa_pairs)

    for qa in qa_pairs:
        question = qa["question"]
        answer = qa["answer"].strip().lower()

        if retrieval_method == "bm25":
            retrieved = retrieve_bm25(chunks, question, top_k=1)
        elif retrieval_method == "dense":
            retrieved = retrieve_dense(chunks, question, top_k=1)
        elif retrieval_method == "hybrid":
            retrieved = retrieve_hybrid(chunks, question, top_k=1)
        elif retrieval_method == "rerank":
            retrieved = retrieve_rerank(chunks, question, top_k=1)
        else:
            retrieved = []

        if retrieved and answer in retrieved[0].lower():
            correct += 1

    return correct / total if total > 0 else 0.0

# ── Process one document ───────────────────────────────────────────────────────
def process_document(txt_path, qa_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    if not text.strip() or not qa_pairs:
        return None

    results = {}
    for config in CONFIGS:
        config_id = config["config_id"]
        chunk_size = config["chunk_size"]
        method = config["retrieval_method"]

        chunks = chunk_text(text, chunk_size)
        if not chunks:
            results[config_id] = 0.0
            continue

        precision = compute_retrieval_precision(chunks, qa_pairs, method)
        results[config_id] = precision

    # Best config = one with highest precision
    best_config = max(results, key=results.get)
    results["best_config"] = best_config

    return results

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_rows = []
    domains = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

    total_docs = 0
    for domain in sorted(domains):
        domain_path = os.path.join(DATA_DIR, domain)
        txt_files = sorted([f for f in os.listdir(domain_path) if f.endswith(".txt")])
        total_docs += len(txt_files)

    print(f"\nTotal documents to process: {total_docs}")
    print(f"Total config runs: {total_docs * len(CONFIGS)}\n")

    with tqdm(total=total_docs, desc="Processing documents") as pbar:
        for domain in sorted(domains):
            domain_path = os.path.join(DATA_DIR, domain)
            txt_files = sorted([f for f in os.listdir(domain_path) if f.endswith(".txt")])

            for txt_file in txt_files:
                base = txt_file.replace(".txt", "")
                txt_path = os.path.join(domain_path, txt_file)
                qa_path = os.path.join(domain_path, f"{base}_qa.json")

                if not os.path.exists(qa_path):
                    print(f"  ⚠ Missing QA file for {txt_file}, skipping.")
                    pbar.update(1)
                    continue

                result = process_document(txt_path, qa_path)

                if result:
                    row = {
                        "filename": txt_file,
                        "domain": domain,
                        **result
                    }
                    all_rows.append(row)

                pbar.update(1)

    if not all_rows:
        print("No results generated. Check your data/ folder.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Done!")
    print(f"   Documents processed : {len(df)}")
    print(f"   Output saved to     : {os.path.abspath(OUTPUT_FILE)}")
    print(f"\nBest config distribution:")

if __name__ == "__main__":
    main()
