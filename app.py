from __future__ import annotations

import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

import rag

load_dotenv()

st.set_page_config(page_title="StudyBuddy · AI Study Assistant", page_icon="🎓", layout="wide")

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_STORED_MESSAGES = 200
STARTERS = [
    "What is the difference between a stack and a queue?",
    "Explain Big-O notation with an example.",
    "How does a primary key differ from a foreign key?",
]


# --------------------------------------------------------------------------- #
# Cached, process-wide resources (loaded once, shared by all sessions)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading embedding model (first run only)…")
def get_embeddings():
    return rag.make_embeddings()


@st.cache_resource(show_spinner="Indexing the built-in Computer Science handbook…")
def get_default_store():
    return rag.load_default_store(get_embeddings())


@st.cache_resource(show_spinner=False)
def get_tutor(api_key: str, model: str) -> rag.Tutor:
    return rag.Tutor(api_key, model)


def resolve_api_key() -> str | None:
    try:
        key = st.secrets.get("GOOGLE_API_KEY")
    except Exception:  # no secrets.toml locally
        key = None
    return key or os.getenv("GOOGLE_API_KEY") or st.session_state.get("api_key_input")


# --------------------------------------------------------------------------- #
# Session state (per browser tab)
# --------------------------------------------------------------------------- #
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("user_store", None)
ss.setdefault("indexed", {})  # sha256 -> {"name", "subject", "chunks"}


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)})"):
        for s in sources:
            page = f", p.{s['page']}" if s.get("page") else ""
            st.markdown(f"**[{s['tag']}] {s['source']}{page}** · _{s['subject']}_")
            st.caption(s["snippet"])


def chat_as_markdown() -> str:
    lines = [f"# StudyBuddy chat · {datetime.now():%Y-%m-%d %H:%M}\n"]
    for m in ss.messages:
        who = "🧑 Student" if m["role"] == "user" else "🎓 StudyBuddy"
        lines.append(f"**{who}:**\n\n{m['content']}\n")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🎓 StudyBuddy")
    st.caption("Personalised AI tutor grounded in *your* material.")

    with st.expander("👤 Personalise", expanded=True):
        name = st.text_input("Your name", value="Student")
        level = st.selectbox("Level", ["Grade 9-10", "Grade 11", "Grade 12", "Undergraduate"], index=2)
        style = st.selectbox("Explanation style", list(rag.STYLE_GUIDE))
        length = st.select_slider("Answer length", list(rag.LENGTH_GUIDE), value="Medium")
        language = st.selectbox("Reply language", ["English", "Urdu", "Hindi", "Arabic", "Spanish", "French"])
        examples = st.toggle("Include worked examples", value=True)
        goal = st.text_input("Learning goal", placeholder="e.g. Ace the board exam on databases")
        mode = st.radio("Study mode", list(rag.MODE_GUIDE), horizontal=True)

    with st.expander("📚 Study material", expanded=True):
        use_default = st.toggle("Use built-in Computer Science handbook", value=True)
        subject_choice = st.selectbox("Subject for new uploads", rag.SUBJECTS)
        custom = st.text_input("Custom subject name") if subject_choice == "Other" else ""
        subject = custom.strip() or subject_choice

        if ss.indexed:
            st.markdown("**Indexed files**")
            for v in ss.indexed.values():
                st.caption(f"• {v['name']} · {v['subject']} · {v['chunks']} chunks")

    available = ([rag.DEFAULT_SUBJECT] if use_default else []) + sorted({v["subject"] for v in ss.indexed.values()})
    subjects = st.multiselect("🔎 Search only in…", available, placeholder="All sources")
    top_k = st.slider("Passages retrieved", 3, 8, rag.TOP_K)

    with st.expander("⚙️ Model / API key"):
        model = st.text_input("Gemini model", value=DEFAULT_MODEL)
        st.text_input("Google API key", type="password", key="api_key_input",
                      help="Only needed if GOOGLE_API_KEY isn't set in secrets/env.")

    c1, c2 = st.columns(2)
    if c1.button("🗑️ Clear chat", use_container_width=True):
        ss.messages = []
        st.rerun()
    c2.download_button("⬇️ Export", chat_as_markdown(), "studybuddy_chat.md",
                       disabled=not ss.messages, use_container_width=True)

    st.caption("LangChain · FAISS · FastEmbed (ONNX) · Gemini · Streamlit")


# --------------------------------------------------------------------------- #
# Main chat area
# --------------------------------------------------------------------------- #
chat_col, upload_col = st.columns([3, 2], gap="large")
with chat_col:
    st.header("Ask your study material")

with upload_col:
    with st.container(border=True):
        st.subheader("Add your study PDF")
        if use_default:
            st.caption("The built-in Computer Science handbook is being used until you add a custom PDF.")
        else:
            st.caption("The built-in handbook is off. Add a custom PDF to use your own study material.")
        files = st.file_uploader(
            f"Upload PDFs (max {rag.MAX_PDF_MB} MB each)", type="pdf", accept_multiple_files=True
        )
        if st.button("➕ Index uploaded PDFs", disabled=not files, use_container_width=True):
            embeddings = get_embeddings()
            bar = st.progress(0.0, text="Starting…")
            for n, f in enumerate(files, 1):
                data = f.getvalue()
                digest = rag.file_digest(data)
                if digest in ss.indexed:
                    st.info(f"Skipped {f.name} (already indexed).")
                    continue
                try:
                    bar.progress(0.0, text=f"Reading {f.name} ({n}/{len(files)})…")
                    docs = rag.load_pdf(data, f.name, subject)
                    ss.user_store, n_chunks = rag.add_to_store(
                        ss.user_store,
                        docs,
                        embeddings,
                        progress=lambda p, nm=f.name: bar.progress(p, text=f"Embedding {nm}…"),
                    )
                    ss.indexed[digest] = {"name": f.name, "subject": subject, "chunks": n_chunks}
                    st.success(f"Indexed {f.name} → {n_chunks} chunks")
                except Exception as exc:
                    st.error(str(exc))
            bar.empty()

stores = []
if use_default:
    stores.append(get_default_store())
if ss.user_store is not None:
    stores.append(ss.user_store)

api_key = resolve_api_key()

if not stores:
    st.info("Enable the built-in handbook or upload and index a PDF to get started.")
if not api_key:
    st.warning("Add `GOOGLE_API_KEY` to Streamlit secrets (or paste a key under ⚙️ in the sidebar).")

if not ss.messages and stores:
    with chat_col:
        st.markdown("**Try a question:**")
        for q in STARTERS:
            if st.button(q, use_container_width=True):
                ss.pending = q

for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_sources(m.get("sources", []))

prompt = st.chat_input("Ask a question, or say a topic for a quiz/summary…", disabled=not (stores and api_key))
prompt = prompt or ss.pop("pending", None)

if prompt and stores and api_key:
    profile = rag.StudentProfile(name, level, style, length, language, examples, goal)
    history = rag.history_to_messages(ss.messages)

    ss.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            tutor = get_tutor(api_key, model)
            with st.spinner("Searching your material…"):
                query = tutor.rewrite(prompt, history)
                docs = rag.retrieve(query, stores, subjects, top_k)
            answer = st.write_stream(tutor.stream(prompt, history, docs, profile, mode))
            sources = rag.format_sources(docs)
            render_sources(sources)
            ss.messages.append({"role": "assistant", "content": answer, "sources": sources})
        except Exception as exc:
            ss.messages.pop()  # don't keep an unanswered question in history
            st.error(f"Something went wrong while answering: {exc}")

    ss.messages = ss.messages[-MAX_STORED_MESSAGES:]
