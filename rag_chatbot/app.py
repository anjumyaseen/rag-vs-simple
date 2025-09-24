# rag_chatbot/app.py
import os
import re
import glob
import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv

# Quiet Chroma telemetry noise before importing chromadb
os.environ.setdefault("CHROMA_TELEMETRY", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3")

HERE = os.path.dirname(__file__)
DOCS_DIR = os.path.abspath(os.path.join(HERE, "..", "docs"))
CHROMA_DIR = os.path.join(HERE, "chroma")

st.set_page_config(page_title="RAG Chatbot (Hybrid Retrieval + Citations)", page_icon="🔎")
st.title("🔎 RAG Chatbot (Hybrid Retrieval + Citations)")
st.caption("Semantic + BM25 retrieval → merged & reranked → answer with citations (local & free).")

# ---------------- Utilities ----------------
def clean_text(t: str) -> str:
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def chunk_words(text: str, chunk_size=250, overlap=40):
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
            break

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
def tokenize(s: str):
    return TOKEN_RE.findall(s.lower())

EMAIL_RE  = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_RE    = re.compile(r"https?://\S+|www\.\S+")
PHONE_RE  = re.compile(r"\+?\d[\d \-().]{7,}\d")

def looks_like_contact(text: str) -> bool:
    return bool(EMAIL_RE.search(text) or URL_RE.search(text) or PHONE_RE.search(text))

def expand_query(q: str) -> str:
    ql = q.lower()
    if any(k in ql for k in ("contact", "contacts", "support", "email", "phone", "website", "help")):
        return q + " contact support email phone website help"
    return q

# ---------------- Caches ----------------
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=False))
    return client.get_collection("docs")

@st.cache_resource
def get_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def build_bm25_index(docs_dir=DOCS_DIR):
    """Build a lightweight BM25 index over chunked docs (fast, in-memory)."""
    texts, ids, metas = [], [], []
    for path in glob.glob(os.path.join(docs_dir, "**/*"), recursive=True):
        if os.path.isdir(path) or not path.lower().endswith((".md", ".txt")):
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = clean_text(f.read())
        fname = os.path.basename(path)
        for ch in chunk_words(raw, chunk_size=250, overlap=40):
            cid = f"{fname}:{ch['idx']}"
            ids.append(cid)
            texts.append(ch["text"])
            metas.append({"doc": fname})
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized) if tokenized else None
    return {"bm25": bm25, "texts": texts, "ids": ids, "metas": metas}

collection = get_collection()
embedder = get_embedder()
bm = build_bm25_index()

# ---------------- Retrieval ----------------
def retrieve_semantic(query: str, k: int = 8):
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["metadatas", "documents", "distances"],  # "ids" not allowed in include
    )
    ids = res.get("ids", [[]])[0] or [str(i) for i in range(len(res["documents"][0]))]
    hits = []
    for i, _id in enumerate(ids):
        score = 1.0 - float(res["distances"][0][i])  # convert distance → similarity-ish
        hits.append({
            "id": _id,
            "doc": res["metadatas"][0][i]["doc"],
            "text": res["documents"][0][i],
            "score_sem": score,
        })
    return hits

def retrieve_bm25(query: str, top_n: int = 20):
    bm25 = bm["bm25"]
    if not bm25:
        return []
    toks = tokenize(query)
    if not toks:
        return []
    scores = bm25.get_scores(toks)                 # numpy array or list
    arr = np.asarray(scores, dtype=float).ravel()
    if arr.size == 0:
        return []
    top_n = int(min(top_n, arr.size))
    idxs = np.argpartition(-arr, range(top_n))[:top_n]
    idxs = idxs[np.argsort(-arr[idxs])]
    maxs = float(arr.max()) if arr.size else 1.0
    hits = []
    for i in idxs:
        i = int(i)
        hits.append({
            "id": bm["ids"][i],
            "doc": bm["metas"][i]["doc"],
            "text": bm["texts"][i],
            "score_lex": (arr[i] / (maxs + 1e-9)),
        })
    return hits

def combine_hits(sem_hits, lex_hits, is_contact_intent: bool, k: int):
    # dedupe by id and merge scores
    merged = {}
    for h in sem_hits:
        merged[h["id"]] = {"id": h["id"], "doc": h["doc"], "text": h["text"],
                           "score_sem": h.get("score_sem", 0.0), "score_lex": 0.0}
    for h in lex_hits:
        m = merged.get(h["id"])
        if m:
            m["score_lex"] = max(m["score_lex"], h.get("score_lex", 0.0))
        else:
            merged[h["id"]] = {"id": h["id"], "doc": h["doc"], "text": h["text"],
                               "score_sem": 0.0, "score_lex": h.get("score_lex", 0.0)}
    hits = list(merged.values())

    # contact-aware boost
    for h in hits:
        h["contact_boost"] = 0.15 if is_contact_intent and looks_like_contact(h["text"]) else 0.0

    # weighted combo (tuneable)
    for h in hits:
        h["score"] = 0.6 * h["score_sem"] + 0.4 * h["score_lex"] + h["contact_boost"]

    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[:k]

def retrieve_hybrid(query: str, k: int, is_contact_intent: bool):
    # adapt K for very short / vague queries
    k_sem = max(k, 8) if len(query.split()) <= 2 else k
    k_lex = max(20, k * 3)
    sem = retrieve_semantic(query, k_sem)
    lex = retrieve_bm25(query, k_lex)
    # If clearly contact intent and nothing looks like contact, do a recall sweep
    if is_contact_intent and not any(looks_like_contact(h["text"]) for h in sem + lex):
        lex = retrieve_bm25(query + " email phone contact support website", top_n=k_lex * 2)
    return combine_hits(sem, lex, is_contact_intent, k)

# ---------------- Generation ----------------
def ollama_chat(system: str, user: str) -> str:
    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return (f"Error: cannot reach Ollama at {OLLAMA_URL}. "
                "Start Ollama and pull a model (e.g., `ollama pull llama3`).")
    except Exception as e:
        return f"Error: {e}"

def make_prompt(query: str, hits):
    ctx_lines = [f"[{h['doc']}] :: {h['text']}" for h in hits]
    context = "\n\n".join(ctx_lines)
    sys = (
        "You are a careful, fact-based assistant.\n"
        "Use ONLY the provided context. If the user asks something broad like 'contacts' or 'support', "
        "list any emails, phone numbers, or websites that appear in the context.\n"
        "If the answer is not in the context, reply exactly: \"I don't know based on these docs.\"\n"
        "Always include short citations like [<filename.md>]."
    )
    user = f"Question: {query}\n\nContext:\n{context}"
    return sys, user

# ---------------- UI ----------------
if "history" not in st.session_state:
    st.session_state.history = []
if "last_hits" not in st.session_state:
    st.session_state.last_hits = []

k = st.sidebar.slider("Top K chunks", 3, 12, 6)
st.sidebar.caption("K controls how many retrieved chunks go to the LLM.")

prompt = st.chat_input("Ask about the docs…")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})

    q_expanded = expand_query(prompt)
    is_contact = any(w in prompt.lower() for w in ("contact","contacts","support","email","phone","website","help"))

    hits = retrieve_hybrid(q_expanded, k, is_contact)
    st.session_state.last_hits = hits  # save for debug panel

    sys, user = make_prompt(prompt, hits)
    answer = ollama_chat(sys, user)

    cites = "\n".join({f"- {h['doc']} (score ~{h['score']:.3f})" for h in hits})
    final = f"{answer}\n\n**Sources:**\n{cites}" if hits else answer
    st.session_state.history.append({"role": "assistant", "content": final})

# Render chat
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Optional debug view of retrieved chunks
with st.expander("Debug: retrieved chunks", expanded=False):
    _hits = st.session_state.get("last_hits", [])
    if _hits:
        for i, h in enumerate(_hits[:k], 1):
            st.markdown(
                f"**{i}. {h['doc']}** "
                f"(score={h.get('score', 0):.3f}, "
                f"sem={h.get('score_sem', 0):.3f}, "
                f"lex={h.get('score_lex', 0):.3f})"
            )
            st.write(h["text"][:500] + ("…" if len(h["text"]) > 500 else ""))
    else:
        st.write("Ask a question to see retrieved chunks here.")
