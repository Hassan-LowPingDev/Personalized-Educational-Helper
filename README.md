# 🎓 StudyBuddy — Personalised RAG Study Assistant

A Streamlit app where students upload their own subject PDFs (physics, maths,
computer science, ...) and chat with a tutor that answers **only from their
material**, with citations, in the style they choose.

**Try it instantly:** a built-in Computer Science handbook is pre-indexed, so
no upload is needed. Click a starter question and watch the cited answer stream in.

## Features
| Feature | How it works |
|---|---|
| Personalisation | Name, level, explanation style, length, language, examples, learning goal, study mode (Q&A / Quiz / Summary) are injected into the system prompt |
| PDF upload | Multi-file, per-subject tagging, SHA-256 de-duplication, size/page limits, clear errors for scanned PDFs |
| Chat memory | Last 6 turns sent to the LLM; a query-rewriting step turns follow-ups ("explain that again") into standalone retrieval queries |
| Citations | Every answer cites `[S1]`, `[S2]`... with source file, page and snippet |
| Subject filter | Restrict retrieval to chosen subjects via FAISS metadata filtering |
| Export | Download the conversation as Markdown |

## Architecture
```
PDF → PyPDFLoader → RecursiveCharacterTextSplitter (900/150)
    → FastEmbed bge-small (ONNX, CPU) → FAISS (in-memory)
User question + history → rewrite → top-k retrieval (+subject filter)
    → prompt (profile + context + history) → Gemini (streaming) → answer + sources
```

## Engineering decisions (built for free-tier Streamlit hosting)
- **FAISS instead of Chroma:** no SQLite version conflicts on Streamlit Cloud.
- **FastEmbed (ONNX) instead of PyTorch sentence-transformers:** ~10x smaller install and far lower RAM; cold start stays fast.
- **`st.cache_resource`** for the embedding model, default index and LLM client: loaded once per process, shared across sessions.
- **Per-session user index** in `st.session_state`, so one visitor's uploads never leak to another.
- **Batched embedding with a progress bar** keeps memory flat and the UI responsive.
- **Streaming responses** (`st.write_stream`) for low perceived latency.
- **Guardrails:** upload size/page caps, de-duplication, graceful error handling, bounded history.
- **Separation of concerns:** `rag.py` has no Streamlit imports and is unit-testable; `app.py` is UI only.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add GOOGLE_API_KEY
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to GitHub (`.env` and `secrets.toml` are git-ignored).
2. On share.streamlit.io choose the repo, main file `app.py`.
3. Under **Advanced settings → Secrets** add: `GOOGLE_API_KEY = "..."` (and optionally `GEMINI_MODEL = "..."`).

## Known limitations / next steps
- Uploaded indexes live in memory for the session (Community Cloud storage is ephemeral). Next step: persist to a managed vector DB.
- Scanned PDFs need OCR (e.g. `ocrmypdf`) before upload.
- Possible additions: reranking, hybrid BM25 search, evaluation set (RAGAS), auth.
