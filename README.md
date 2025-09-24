# RAG vs Simple Chatbot (Ollama, Local & Free)

Two side-by-side apps over the same docs:

- **A. Simple Chatbot** — just dumps docs into the prompt (OK for tiny corpora)
- **B. RAG Chatbot** — chunk → embed → retrieve → answer with citations (scales)

---

## 1) Prerequisites

- **Python** 3.10+
- **Ollama** installed: https://ollama.com/download  
  After install, pull a model:
  ```bash
  ollama pull llama3
  # or: ollama pull mistral
Ollama REST API runs at http://localhost:11434 by default.