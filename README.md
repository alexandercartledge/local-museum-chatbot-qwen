# local-museum-chatbot-qwen

Local museum chatbot built with FastAPI, FAISS and Qwen via Ollama.  
It performs room level retrieval over exhibit texts, answers Italian and English questions using a fully local RAG pipeline, and includes a simple embeddable web UI.

## What this project does

- Loads curated museum texts and groups them by room.
- Builds a FAISS index using SentenceTransformers for semantic search.
- Selects the most relevant room for each visitor question.
- Calls a local Qwen model through Ollama, grounded only on that room text.
- Serves a small chat page that can be opened directly or embedded in another site.

## Main pieces in this repo

- `app/server.py` FastAPI backend that exposes a `/ask` endpoint and uses Qwen + FAISS.
- `app/ingest.py` Script that reads `data/chunks.csv` and builds `index/faiss.index` and `meta.pkl`.
- `web/embed.html` Minimal HTML and JavaScript chat widget that talks to the backend.
- `run.bat` Helper script for starting the server on Windows.
- `.env` Example configuration for model names, index directory and Ollama URL.

Note: this public repo only contains the code. Real museum texts and the generated FAISS index are not included.
