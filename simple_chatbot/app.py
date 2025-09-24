import os
import glob
import textwrap
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# --- Robust docs path ---
HERE = os.path.dirname(__file__)
DEFAULT_DOCS_DIR = os.path.abspath(os.path.join(HERE, "..", "docs"))
DOCS_DIR = os.environ.get("DOCS_DIR", DEFAULT_DOCS_DIR)

st.set_page_config(page_title="Simple Chatbot (No RAG)", page_icon="💬")
st.title("💬 Simple Chatbot (No RAG)")
st.caption("Loads entire docs into the prompt (OK for tiny corpora; not scalable).")

def load_docs_text(docs_dir=DOCS_DIR):
    parts = []
    for path in glob.glob(os.path.join(docs_dir, "**/*"), recursive=True):
        if os.path.isdir(path):
            continue
        if path.lower().endswith((".md", ".txt")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                parts.append(f"\n\n=== FILE: {os.path.basename(path)} ===\n{f.read()}")
    return "\n".join(parts)[:100_000]

# Load once
DOCS_TEXT = load_docs_text()

# Debug panel so you can see where it's looking
with st.sidebar.expander("Docs debug", expanded=False):
    st.write(f"Docs dir: `{DOCS_DIR}`")
    files = [p for p in glob.glob(os.path.join(DOCS_DIR, '**/*'), recursive=True)
             if os.path.isfile(p) and p.lower().endswith(('.md', '.txt'))]
    if files:
        st.write("Found files:")
        for p in files:
            st.write("•", os.path.basename(p))
    else:
        st.warning("No `.md` or `.txt` files found here.")

if not DOCS_TEXT:
    st.warning(f"No docs found under `{DOCS_DIR}`. Add some `.md` or `.txt` files.")

def ollama_chat(messages):
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": MODEL, "stream": False, "messages": messages},
            timeout=120,
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
