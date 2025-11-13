# src/retrieval.py
"""
Retriever that loads a FAISS index + meta.pkl and encodes queries.
Default: uses sentence-transformers SBERT (recommended since your FAISS index was built with SBERT).
Optional: set EMBED_PROVIDER=openrouter to use OpenRouter embeddings for query encoding.
"""

import os
import faiss
import pickle
import numpy as np
from typing import List

INDEX_FILE = "vectors.faiss"
META_FILE = "meta.pkl"

# OpenRouter settings (adjust model name if needed)
OPENROUTER_EMBED_MODEL = "text-embedding-3-small"  # or the model name shown in your OpenRouter dashboard
OPENROUTER_URL_EMB = "https://api.openrouter.ai/v1/embeddings"

class Retriever:
    def __init__(self, sbert_model_name: str = "all-MiniLM-L6-v2"):
        if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
            raise FileNotFoundError(f"Missing {INDEX_FILE} or {META_FILE}. Run embed_index.py first.")

        self.index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "rb") as f:
            self.meta = pickle.load(f)
        n = len(self.meta[next(iter(self.meta))])
        self.records = [{k: self.meta[k][i] for k in self.meta} for i in range(n)]

        # SBERT (preferred, since index built with SBERT)
        self.sbert = None
        try:
            from sentence_transformers import SentenceTransformer
            self.sbert = SentenceTransformer(sbert_model_name)
            print(f"[retriever] SBERT ready ('{sbert_model_name}').")
        except Exception as e:
            print(f"[retriever] SBERT not available: {e}")

        # OpenRouter config
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.embed_provider = os.getenv("EMBED_PROVIDER", "sbert").lower()

        if self.embed_provider not in ("sbert", "openrouter"):
            raise ValueError("EMBED_PROVIDER must be 'sbert' or 'openrouter'")

    # ---------------- SBERT embedding ----------------
    def _embed_with_sbert(self, text: str) -> np.ndarray:
        emb = self.sbert.encode([text], convert_to_numpy=True)
        return emb.astype("float32")

    # ---------------- OpenRouter embedding ----------------
    def _embed_with_openrouter(self, text: str) -> np.ndarray:
        # requires OPENROUTER_API_KEY set
        if not self.openrouter_key:
            raise RuntimeError("OPENROUTER_API_KEY not set for OpenRouter embeddings.")
        import requests
        payload = {"model": OPENROUTER_EMBED_MODEL, "input": [text]}
        headers = {"Authorization": f"Bearer {self.openrouter_key}", "Content-Type": "application/json"}
        r = requests.post(OPENROUTER_URL_EMB, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        j = r.json()
        # expects j["data"][0]["embedding"]
        emb = np.array(j["data"][0]["embedding"], dtype="float32").reshape(1, -1)
        return emb

    # ---------------- encode query (dispatch) ----------------
    def encode_query(self, query: str) -> np.ndarray:
        if self.embed_provider == "sbert":
            if self.sbert is None:
                raise RuntimeError("SBERT not available; install sentence-transformers or set EMBED_PROVIDER=openrouter")
            return self._embed_with_sbert(query)
        else:
            # openrouter
            return self._embed_with_openrouter(query)

    # ---------------- retrieval ----------------
    def retrieve(self, query: str, k: int = 5) -> List[dict]:
        q_emb = self.encode_query(query)
        distances, idxs = self.index.search(q_emb, k)
        idxs = idxs[0].tolist()
        distances = distances[0].tolist()
        hits = []
        for i, d in zip(idxs, distances):
            if i < 0 or i >= len(self.records):
                continue
            rec = dict(self.records[i])
            rec["_score"] = float(d)
            hits.append(rec)
        return hits

if __name__ == "__main__":
    r = Retriever()
    print(r.retrieve("Which product category had the highest sales last quarter?", k=3))
