# RAG vs Simple (Local & Free) 🧪

Two side-by-side chat apps over the **same docs** so you can feel the difference:

- **Simple Chatbot** — pastes your docs straight into the prompt (OK for tiny corpora).
- **RAG Chatbot (Hybrid)** — chunk → embed → **hybrid retrieval** (semantic + BM25) → answer with **citations**. Scales better.

Runs **fully local** using [Ollama](https://ollama.com) (no paid API keys).

---

## ✨ Purpose

This project is a learning + starter kit to compare a naive “paste-all-docs” chatbot with a production-style **RAG** (Retrieval Augmented Generation) pipeline. You’ll see when a simple prompt works, when it fails, and how RAG fixes it with **chunking, embeddings, retrieval, and citations**.

---

## ✨ What you’ll get

- 100% local: Ollama for the LLM; nothing leaves your machine.
- RAG pipeline with **Chroma** vector DB + **sentence-transformers** embeddings.
- **Hybrid retrieval**: semantic (embeddings) + lexical (BM25) merged & re-ranked.
- Handles **vague queries** like “contacts?” via query expansion & retrieval tweaks.
- Streamlit UI with **Top-K slider** and a retrieval **debug** panel.
- Minimal, hackable codebase designed for learning.

---

## 🧱 Project Structure

```
rag-vs-simple/
├── README.md
├── .gitignore
├── .env.example
├── requirements.txt
├── requirements.cpu.txt                # CPU-only PyTorch (prevents CUDA downloads)
├── sitecustomize.py                    # use modern SQLite for Chroma on older systems
├── docs/
│   ├── sample_intro.md
│   └── product_faq.md
├── simple_chatbot/
│   └── app.py
└── rag_chatbot/
    ├── ingest.py
    └── app.py
```

> The **RAG index** is stored under `rag_chatbot/chroma/` (git-ignored).

---

## ✅ Requirements (What you need)

- **Python 3.10+**
- **Ollama** installed: <https://ollama.com/download>
  ```bash
  ollama pull llama3   # or mistral / qwen2.5 / llama3.1:8b, etc.
  ```
- (Windows users) **WSL2** is recommended. Keep the project on the **WSL filesystem** (e.g., `~/projects/...`) for much faster I/O than `/mnt/c/...`.

---

## 🚀 Quickstart (copy–paste)

```bash
git clone <your-fork-url> rag-vs-simple
cd rag-vs-simple

python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux/WSL:  source .venv/bin/activate

# 1) Install CPU-only PyTorch first (avoids big NVIDIA/CUDA wheels)
pip install -r requirements.cpu.txt

# 2) Install app dependencies
pip install -r requirements.txt
```

Create `.env` (optional — defaults are fine):
```bash
cp .env.example .env
# OLLAMA_URL=http://localhost:11434
# OLLAMA_MODEL=llama3
```

> If you see `ModuleNotFoundError: rank_bm25`, run `pip install rank-bm25` (some older zips may not include it in requirements).

---

## ▶️ Run the Apps

### A) Simple Chatbot (No RAG)
```bash
streamlit run simple_chatbot/app.py --server.port 8501
# open http://localhost:8501
```

### B) Build the RAG Index, then Run the RAG Chatbot
```bash
python rag_chatbot/ingest.py
streamlit run rag_chatbot/app.py --server.port 8502
# open http://localhost:8502
```

> Re-run `ingest.py` whenever you **add/edit** files in `./docs/`.

---

## 🧪 What to Ask (Demo)

**Works in both apps:**
- “What services do we provide?”
- “Is Password Safe cloud-based?”
- “What guarantees do we offer?”
- “What’s the support email and website?”

**Vague queries (Hybrid RAG handles better):**
- “contacts?”
- “support?”
- “how to reach you?”

**Update test:**
1. Edit `docs/product_faq.md` (e.g., change the email).
2. Run `python rag_chatbot/ingest.py`.
3. Ask again; answers update with the docs.

---

## ⚙️ Configuration

`.env` (example values shown):
```ini
# Ollama HTTP endpoint
OLLAMA_URL=http://localhost:11434

# Any downloaded Ollama model
OLLAMA_MODEL=llama3
```

**Advanced (optional):**
- `DOCS_DIR` (Simple app only): override the docs folder it scans.
- `CHROMA_DIR` (ingest): override where the Chroma DB is stored (defaults to `rag_chatbot/chroma`).

---

## 🧠 How It Works (High Level)

```
User question
   │
   ├── Simple Chatbot → paste entire docs into the prompt → LLM answers (filename citation)
   │
   └── RAG Chatbot
        ├─ Ingest: docs → clean → chunk → embed → Chroma (persistent, local)
        ├─ Query: semantic (Chroma) + BM25 (in-memory) → merged & re-ranked (contact-aware boost)
        └─ Prompt LLM with Top-K chunks → Answer + citations
```

**Top-K slider:** Controls how many retrieved chunks are sent to the LLM.  
- Lower K (3–4) → precise, faster.  
- Higher K (8–12) → more recall (useful for broad questions), but slower/wordier.

---

## 🛠 Troubleshooting

**Ollama connection error in the UI**  
- Message: `Error: cannot reach Ollama at http://localhost:11434`  
- Fix:
  ```bash
  ollama pull llama3
  curl http://localhost:11434/api/tags
  ```
  On Windows+WSL, if Ollama runs on Windows, point WSL to the Windows host IP:
  ```bash
  WINIP=$(grep nameserver /etc/resolv.conf | awk '{print $2}')
  export OLLAMA_URL="http://$WINIP:11434"
  ```

**Streamlit prints `gio: http://localhost:8501: Operation not supported`**  
- Harmless: Streamlit tried to auto-open a browser in WSL. Open the URL manually.

**Chroma error: `sqlite3 >= 3.35.0` required**  
- This repo includes `sitecustomize.py` to make Python import `pysqlite3-binary` as `sqlite3` (modern SQLite).  
- If needed, install/upgrade:
  ```bash
  pip install -U pysqlite3-binary
  ```

**Ingest slow / telemetry spam**  
- Telemetry logs are harmless. Silence for this shell:
  ```bash
  export CHROMA_TELEMETRY=false ANONYMIZED_TELEMETRY=false
  ```
- Keep the project (and especially the index path) on **WSL filesystem** (e.g., `~/projects/...`), not `/mnt/c`.

**Linux/WSL prints `Killed` during ingest**  
- That’s the kernel OOM-killer. Use smaller batches in `ingest.py` (e.g., `build_index(batch_size=4)`), or increase WSL RAM via `C:\Users\<You>\.wslconfig`:
  ```ini
  [wsl2]
  memory=8GB
  swap=8GB
  ```

**Index duplicates / huge collection counts**  
- We ship a terminating chunker and idempotent upserts. To hard reset:
  ```bash
  rm -rf rag_chatbot/chroma
  python rag_chatbot/ingest.py
  ```

---

## 🧩 Extending Ideas

- **Reranker**: plug in a cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to rescore hits.
- **PDFs**: add a PDF loader in `ingest.py` (we already include `pypdf`).
- **Eval**: create a small Q/A set to compare Simple vs RAG accuracy.
- **Auth/Logging**: add user auth and store feedback on answers.

---

## 🧾 License

MIT (change to your preferred license if needed).
