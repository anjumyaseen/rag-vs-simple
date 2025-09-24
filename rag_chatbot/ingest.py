# rag_chatbot/ingest.py
import os
import glob
import re
from typing import List, Dict, Iterable

# Silence telemetry noise
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("CHROMA_TELEMETRY", "false")

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Paths
ROOT = os.path.dirname(__file__)
DOCS_DIR = os.path.abspath(os.path.join(ROOT, "..", "docs"))
CHROMA_DIR = os.environ.get("CHROMA_DIR", os.path.join(ROOT, "chroma"))

# ------------------ loaders & chunking ------------------

def read_text_files(docs_dir: str) -> List[Dict]:
    items = []
    for path in glob.glob(os.path.join(docs_dir, "**/*"), recursive=True):
        if os.path.isdir(path):
            continue
        if path.lower().endswith((".md", ".txt")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                items.append({"name": os.path.basename(path), "text": f.read()})
    return items

def clean_text(t: str) -> str:
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def chunk_words(text: str, chunk_size=250, overlap=40) -> Iterable[Dict]:
    """Guaranteed-terminating chunker."""
    words = text.split()
    L = len(words)
    if L == 0:
        return
    step = max(1, chunk_size - overlap)
    idx = 0
    for start in range(0, L, step):
        end = min(L, start + chunk_size)
        yield {"idx": idx, "text": " ".join(words[start:end])}
        idx += 1
        if end == L:
            break  # stop at the end

# ------------------ ingest in small batches ------------------

def build_index(batch_size: int = 8):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=True))

    # Start from a clean slate for the demo
    client.reset()
    coll = client.create_collection("docs")

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    items = read_text_files(DOCS_DIR)
    print(f"Found {len(items)} files in {DOCS_DIR}")
    total_chunks = 0

    for it in items:
        raw = clean_text(it["text"])
        chunks = list(chunk_words(raw, chunk_size=250, overlap=40))
        print(f"{it['name']}: {len(raw.split())} words -> {len(chunks)} chunks")

        # stream in micro-batches
        buffer_ids, buffer_texts, buffer_metas = [], [], []
        for ch in chunks:
            cid = f"{it['name']}:{ch['idx']}"
            buffer_ids.append(cid)
            buffer_texts.append(ch["text"])
            buffer_metas.append({"doc": it["name"]})

            if len(buffer_texts) >= batch_size:
                embs = embedder.encode(
                    buffer_texts,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                coll.upsert(ids=buffer_ids, embeddings=embs, metadatas=buffer_metas, documents=buffer_texts)
                total_chunks += len(buffer_texts)
                print(f"Added {total_chunks} chunks...", flush=True)
                buffer_ids, buffer_texts, buffer_metas = [], [], []

        # flush remainder
        if buffer_texts:
            embs = embedder.encode(
                buffer_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            coll.upsert(ids=buffer_ids, embeddings=embs, metadatas=buffer_metas, documents=buffer_texts)
            total_chunks += len(buffer_texts)
            print(f"Added {total_chunks} chunks...", flush=True)

    print(f"Done. Indexed {total_chunks} chunks into {CHROMA_DIR}/")

if __name__ == "__main__":
    build_index(batch_size=8)
