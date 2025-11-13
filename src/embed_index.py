# src/embed_index.py
"""
Robust embed_index with automatic fallback to sentence-transformers if OpenAI quota/exceptions occur.

Usage:
    # full run (tries OpenAI then fallback to SBERT if needed)
    python src/embed_index.py

    # test only first 200 rows (cheap, quick)
    python src/embed_index.py --sample 200
"""

import os
import time
import pickle
import argparse
from typing import List
import numpy as np
import faiss
import sys

from data_prep import load_all

# Config
OPENAI_EMBED_MODEL = "text-embedding-3-small"
SBERT_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE_OPENAI = 64
BATCH_SIZE_SBERT = 64
INDEX_FILE = "vectors.faiss"
META_FILE = "meta.pkl"

# ---- OpenAI helper (new client) ----
def embed_with_openai(texts: List[str], model=OPENAI_EMBED_MODEL, batch_size=BATCH_SIZE_OPENAI, retry_max=4):
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package not available. Install openai>=1.0 or use SBERT fallback.") from e

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or None)
    embeddings = []
    n = len(texts)

    for i in range(0, n, batch_size):
        batch = texts[i:i+batch_size]
        attempt = 0
        while True:
            try:
                resp = client.embeddings.create(model=model, input=batch)
                # response.data is list-like with .embedding attribute
                batch_embs = [d.embedding for d in resp.data]
                embeddings.extend(batch_embs)
                print(f"[openai] Embedded {min(i+batch_size,n)}/{n}")
                break
            except Exception as e:
                attempt += 1
                msg = str(e)
                # detect quota errors or 429 quickly
                if "insufficient_quota" in msg or "429" in msg or "quota" in msg.lower():
                    # explicit signal to caller to fallback
                    raise RuntimeError("OPENAI_INSUFFICIENT_QUOTA: " + msg)
                if attempt > retry_max:
                    raise
                wait = 2 ** attempt
                print(f"[openai warn] attempt {attempt}/{retry_max} failed: {e}. retrying in {wait}s...")
                time.sleep(wait)
    return np.array(embeddings, dtype="float32")

# ---- SBERT helper ----
def embed_with_sbert(texts: List[str], model_name=SBERT_MODEL, batch_size=BATCH_SIZE_SBERT):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers not installed. Run: pip install sentence-transformers") from e

    model = SentenceTransformer(model_name)
    embs = model.encode(texts, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)
    return embs.astype("float32")

# ---- FAISS build & metadata ----
def build_faiss_and_save(embeddings: np.ndarray, index_file=INDEX_FILE):
    dim = embeddings.shape[1]
    print(f"[faiss] building index dim={dim} n={embeddings.shape[0]}")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_file)
    print(f"[faiss] saved to {index_file}")

def save_meta(df, meta_file=META_FILE):
    meta = df.reset_index().to_dict(orient="list")
    with open(meta_file, "wb") as f:
        pickle.dump(meta, f)
    print(f"[meta] saved to {meta_file}")

# ---- main ----
def main(sample: int = None, prefer_openai: bool = True):
    print("[1/4] Loading data...")
    df = load_all()
    if sample:
        df = df.iloc[:sample].copy()
        print(f"[info] Running sample mode: using first {len(df)} rows")
    texts = df["text"].astype(str).tolist()

    embeddings = None
    # Try OpenAI first (if requested)
    if prefer_openai:
        try:
            print("[2/4] Trying OpenAI embeddings...")
            embeddings = embed_with_openai(texts)
            print("[info] OpenAI embeddings succeeded.")
        except RuntimeError as e:
            msg = str(e)
            if "OPENAI_INSUFFICIENT_QUOTA" in msg or "insufficient_quota" in msg or "quota" in msg.lower():
                print("[warn] OpenAI quota/429 issue detected. Falling back to sentence-transformers locally.")
            else:
                print(f"[warn] OpenAI embeddings failed with error: {e}. Falling back to sentence-transformers.")
            embeddings = None
        except Exception as e:
            print(f"[warn] OpenAI embeddings unexpected error: {e}. Falling back to SBERT.")
            embeddings = None

    # If embeddings not created yet, use SBERT fallback
    if embeddings is None:
        print("[2/4] Using sentence-transformers (local) fallback...")
        embeddings = embed_with_sbert(texts)
        print("[info] SBERT embeddings created.")

    # Build FAISS and save meta
    print("[3/4] Building FAISS index...")
    build_faiss_and_save(embeddings, INDEX_FILE)

    print("[4/4] Saving metadata...")
    save_meta(df, META_FILE)

    print("\nDone. Created:", INDEX_FILE, META_FILE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="If set, only embed first N rows (for testing).")
    parser.add_argument("--no-openai", action="store_true", help="Force disable OpenAI; use SBERT only.")
    args = parser.parse_args()
    main(sample=args.sample, prefer_openai=not args.no_openai)
