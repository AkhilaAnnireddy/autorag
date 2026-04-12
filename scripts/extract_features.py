import os
import json
import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import random

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
OUTPUT_DIR = "features"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "features.csv")
SBERT_MODEL = "all-MiniLM-L6-v2"
SAMPLE_PARAGRAPHS = 20  # number of paragraphs to sample for SBERT embedding
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Load models ────────────────────────────────────────────────────────────────
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2_000_000  # allow large documents

print("Loading Sentence-BERT model...")
sbert = SentenceTransformer(SBERT_MODEL)

# ── Surface feature extraction ─────────────────────────────────────────────────
def extract_surface_features(text):
    """Extract 7 surface and structural features from document text."""

    # Paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # spaCy doc for tokenization and sentence splitting
    doc = nlp(text[:1_000_000])  # cap at 1M chars for safety

    # 1. Average sentence length (tokens per sentence)
    sentences = list(doc.sents)
    if sentences:
        avg_sent_length = np.mean([len(sent) for sent in sentences])
    else:
        avg_sent_length = 0.0

    # 2. Average paragraph length (tokens per paragraph)
    if paragraphs:
        para_lengths = [len(p.split()) for p in paragraphs]
        avg_para_length = np.mean(para_lengths)
    else:
        avg_para_length = 0.0

    # 3. Total corpus tokens
    total_tokens = len(doc)

    # 4. Vocabulary richness (type-token ratio)
    words = [token.text.lower() for token in doc if token.is_alpha]
    if words:
        vocab_richness = len(set(words)) / len(words)
    else:
        vocab_richness = 0.0

    # 5. Header density (headers per 1000 tokens)
    lines = text.split("\n")
    header_count = sum(
        1 for line in lines
        if line.strip().startswith("#") or
        (line.strip().isupper() and len(line.strip()) > 3)
    )
    header_density = (header_count / total_tokens * 1000) if total_tokens > 0 else 0.0

    # 6. Bullet point density (bullets per 1000 tokens)
    bullet_count = sum(
        1 for line in lines
        if line.strip().startswith(("-", "*", "•", "·")) or
        (len(line.strip()) > 2 and line.strip()[0].isdigit() and line.strip()[1] in (".", ")"))
    )
    bullet_density = (bullet_count / total_tokens * 1000) if total_tokens > 0 else 0.0

    # 7. Table density (table indicators per 1000 tokens)
    table_count = sum(
        1 for line in lines
        if "|" in line or "\t" in line
    )
    table_density = (table_count / total_tokens * 1000) if total_tokens > 0 else 0.0

    return {
        "avg_sent_length": avg_sent_length,
        "avg_para_length": avg_para_length,
        "total_tokens": total_tokens,
        "vocab_richness": vocab_richness,
        "header_density": header_density,
        "bullet_density": bullet_density,
        "table_density": table_density,
    }

# ── SBERT embedding ────────────────────────────────────────────────────────────
def extract_sbert_embedding(text):
    """Sample 20 paragraphs and compute mean SBERT embedding (384-dim)."""
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]

    if not paragraphs:
        # fallback: split into chunks of 200 chars
        paragraphs = [text[i:i+200] for i in range(0, len(text), 200)]

    # Sample up to SAMPLE_PARAGRAPHS paragraphs
    sampled = random.sample(paragraphs, min(SAMPLE_PARAGRAPHS, len(paragraphs)))

    # Encode and mean pool
    embeddings = sbert.encode(sampled, show_progress_bar=False)
    mean_embedding = np.mean(embeddings, axis=0)  # shape: (384,)

    return mean_embedding

# ── Process all documents ──────────────────────────────────────────────────────
def process_all_documents():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows = []
    domains = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

    for domain in sorted(domains):
        domain_path = os.path.join(DATA_DIR, domain)
        txt_files = sorted([f for f in os.listdir(domain_path) if f.endswith(".txt")])

        print(f"\n{'='*60}")
        print(f"Processing domain: {domain} ({len(txt_files)} documents)")
        print(f"{'='*60}")

        for txt_file in tqdm(txt_files, desc=domain):
            filepath = os.path.join(domain_path, txt_file)

            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            if not text.strip():
                print(f"  ⚠ Skipping empty file: {txt_file}")
                continue

            # Extract surface features
            surface = extract_surface_features(text)

            # Extract SBERT embedding
            embedding = extract_sbert_embedding(text)

            # Build row
            row = {
                "filename": txt_file,
                "domain": domain,
                **surface,
            }

            # Add embedding dimensions as sbert_0, sbert_1, ..., sbert_383
            for i, val in enumerate(embedding):
                row[f"sbert_{i}"] = val

            all_rows.append(row)

    return all_rows

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Starting feature extraction...")
    print(f"Data directory: {os.path.abspath(DATA_DIR)}")
    print(f"Output file: {os.path.abspath(OUTPUT_FILE)}")

    rows = process_all_documents()

    if not rows:
        print("No documents found. Check your data/ folder structure.")
        return

    df = pd.DataFrame(rows)

    # Verify dimensions
    feature_cols = [c for c in df.columns if c not in ["filename", "domain"]]
    print(f"Feature extraction complete!")
    print(f"   Documents processed : {len(df)}")
    print(f"   Feature dimensions  : {len(feature_cols)} (expected 391)")
    print(f"   Domains found       : {df['domain'].nunique()}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {os.path.abspath(OUTPUT_FILE)}")

    # Quick summary
    print("\nDocument count per domain:")
    print(df["domain"].value_counts().to_string())

if __name__ == "__main__":
    main()
