import os
import glob
import textwrap
import requests
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").strip()
MODEL = os.getenv("OLLAMA_MODEL", "llama3").strip()
try:
    OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
except ValueError:
    OLLAMA_TIMEOUT = 120.0

# --- Robust docs path ---
HERE = os.path.dirname(__file__)
DEFAULT_DOCS_DIR = os.path.abspath(os.path.join(HERE, "..", "docs"))
DOCS_DIR = os.environ.get("DOCS_DIR", DEFAULT_DOCS_DIR)

SUPPORTED_EXTS = (".md", ".txt", ".pdf")


def _read_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                parts.append(text)
        return "\n".join(parts)
    except Exception as exc:
        st.warning(f"Failed to read PDF '{os.path.basename(path)}': {exc}")
        return ""

st.set_page_config(page_title="Simple Chatbot (No RAG)", page_icon="💬")
st.title("💬 Simple Chatbot (No RAG)")
st.caption("Loads entire docs into the prompt (OK for tiny corpora; not scalable).")

def load_docs_text(docs_dir=DOCS_DIR):
    parts = []
    for path in glob.glob(os.path.join(docs_dir, "**/*"), recursive=True):
        if os.path.isdir(path):
            continue
        lower = path.lower()
        if not lower.endswith(SUPPORTED_EXTS):
            continue
        if lower.endswith(".pdf"):
            text = _read_pdf(path)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        if not text.strip():
            continue
        parts.append(f"\n\n=== FILE: {os.path.basename(path)} ===\n{text}")
    return "\n".join(parts)[:100_000]

# Load once
DOCS_TEXT = load_docs_text()

# Debug panel so you can see where it's looking
with st.sidebar.expander("Docs debug", expanded=False):
    st.write(f"Docs dir: `{DOCS_DIR}`")
    files = [p for p in glob.glob(os.path.join(DOCS_DIR, '**/*'), recursive=True)
             if os.path.isfile(p) and p.lower().endswith(SUPPORTED_EXTS)]
    if files:
        st.write("Found files:")
        for p in files:
            st.write("•", os.path.basename(p))
    else:
        st.warning("No `.md`, `.txt`, or `.pdf` files found here.")

if not DOCS_TEXT:
    st.warning(f"No docs found under `{DOCS_DIR}`. Add some `.md`, `.txt`, or `.pdf` files.")

def ollama_chat(messages):
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": MODEL, "stream": False, "messages": messages},
            timeout=OLLAMA_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return (
            f"Error: cannot reach Ollama at {OLLAMA_URL}. "
            "Start Ollama and pull a model, e.g. `ollama pull llama3`."
        )
    except Exception as e:
        return f"Error: {e}"

if "history" not in st.session_state:
    st.session_state.history = []

user_msg = st.chat_input("Ask about the docs…")
if user_msg:
    user_msg = user_msg.strip()
    st.session_state.history.append({"role": "user", "content": user_msg})

    if user_msg.lower() in {"hi", "hello", "hey"}:
        st.session_state.history.append({"role": "assistant", "content": "Hi! 👋 Ask me about the docs."})
    else:
        system = textwrap.dedent(f"""
        You are a helpful assistant. Answer ONLY using the documentation below.
        If the answer is not in the docs, say "I don't know based on these docs."
        Provide short citations by referencing FILE names when possible.

        --- DOCUMENTATION START ---
        {DOCS_TEXT}
        --- DOCUMENTATION END ---
        """)
        messages = [{"role": "system", "content": system}] + st.session_state.history
        answer = ollama_chat(messages)
        st.session_state.history.append({"role": "assistant", "content": answer})

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
