#!/usr/bin/env python3
import os
import csv
import pickle

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Base directory = repository root (one level above app/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR  = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
INDEX_DIR = os.getenv("INDEX_DIR", os.path.join(BASE_DIR, "index"))
MODEL     = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

os.makedirs(INDEX_DIR, exist_ok=True)

chunks_csv = os.path.join(DATA_DIR, "chunks.csv")
meta_out   = os.path.join(INDEX_DIR, "meta.pkl")
index_out  = os.path.join(INDEX_DIR, "faiss.index")

records, texts = [], []

with open(chunks_csv, encoding="utf-8-sig", newline="") as f:
    r = csv.DictReader(f)
    for row_idx, row in enumerate(r, start=1):
        # Try to read Italian text from either "text_it" or generic "text"
        text_it = (row.get("text_it") or row.get("text") or "").strip()
        if not text_it:
            print(f"Row {row_idx}: empty Italian text, skipping")
            continue

        # NO length filter anymore – we trust your CSV
        # print(f"Row {row_idx}: loaded {len(text_it)} chars")

        rec = {
            "chunk_id": (row.get("chunk_id") or f"chunk_{row_idx}").strip(),
            "scope_type": (row.get("scope_type") or "room").strip(),
            "scope_id": (row.get("scope_id") or "").strip(),
            "url": (row.get("url") or "").strip(),
            "heading": (row.get("heading") or "").strip(),
            "text_it": text_it,
        }

        text_en = (row.get("text_en") or "").strip()
        if text_en:
            rec["text_en"] = text_en

        records.append(rec)
        texts.append(text_it)

print(f"Loaded {len(records)} chunks")

if not texts:
    raise RuntimeError(
        "No valid chunks read from chunks.csv.\n"
        "Check that the file has a 'text_it' or 'text' column with non-empty content."
    )

model = SentenceTransformer(MODEL)
emb = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
emb = np.asarray(emb, dtype=np.float32)

index = faiss.IndexFlatIP(emb.shape[1])  # cosine via normalized vectors
index.add(emb)

faiss.write_index(index, index_out)
with open(meta_out, "wb") as f:
    pickle.dump({"records": records}, f)

print(f"Wrote index → {index_out}\nWrote meta → {meta_out}")
