"""StudyBuddy RAG engine.

Design goals: low memory (no PyTorch), fast cold start, no native-DB quirks
on Streamlit Community Cloud, and clear separation from the UI layer.

Pipeline
--------
PDF -> PyPDFLoader -> RecursiveCharacterTextSplitter -> FastEmbed (ONNX, CPU)
    -> FAISS (in-memory) -> metadata-filtered similarity search
    -> history-aware query rewriting -> Gemini (streaming) with citations
"""
from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # ~130 MB ONNX, 384-dim, strong for its size
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
EMBED_BATCH = 64            # chunks embedded per batch (keeps RAM flat + shows progress)
TOP_K = 5
HISTORY_TURNS = 6           # last N user/assistant pairs sent to the LLM
MAX_PDF_MB = 25
MAX_PAGES = 600             # per PDF; protects free-tier memory/latency

DEFAULT_SUBJECT = "Computer Science (Demo)"
DEFAULT_SOURCE = "Built-in CS Handbook"
DEFAULT_CONTENT_PATH = Path(__file__).parent / "content" / "computer_science.md"
SUBJECTS = ["Computer Science", "Physics", "Mathematics", "Chemistry", "Biology", "English", "Other"]


# --------------------------------------------------------------------------- #
# Personalization
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StudentProfile:
    name: str = "Student"
    level: str = "Grade 12"
    style: str = "Simple & friendly"
    length: str = "Medium"
    language: str = "English"
    examples: bool = True
    goal: str = ""


STYLE_GUIDE = {
    "Simple & friendly": "Use plain words, short sentences and a warm, encouraging tone.",
    "Detailed & technical": "Be precise and rigorous; define terminology and explain the reasoning step by step.",
    "Exam-oriented": "Structure answers like model exam answers: definition, key points, then a one-line summary.",
    "Socratic": "Guide the student with leading questions before revealing the full answer.",
}
LENGTH_GUIDE = {
    "Short": "Keep it under ~100 words.",
    "Medium": "Aim for roughly 150-250 words.",
    "Long": "Be thorough; 300+ words is fine if the topic needs it.",
}
MODE_GUIDE = {
    "Q&A": "Answer the student's question directly.",
    "Quiz me": (
        "Create 5 multiple-choice questions (A-D) on the student's topic using ONLY the context. "
        "Put the answer key with one-line explanations at the very end."
    ),
    "Summary notes": "Produce concise revision notes: headings, bullet points, key formulas/definitions.",
}

SYSTEM_PROMPT = """You are StudyBuddy, a patient AI tutor.

STUDENT PROFILE
- Name: {name}
- Level: {level}
- Explanation style: {style}
- Answer length: {length}
- Reply language: {language}
- Learning goal: {goal}
- Worked examples: {examples}

TASK MODE
{mode_instruction}

RULES
1. Ground every factual statement in the CONTEXT and cite its tag, e.g. [S1], [S2].
2. If the context does not contain the answer, say so plainly. You may then add a brief general
   explanation, clearly labelled "Outside your uploaded material".
3. Never invent page numbers, quotes or sources.
4. Use the conversation history to resolve follow-ups such as "explain that again".
5. Format with Markdown; write formulas in LaTeX ($...$).

CONTEXT
{context}
"""

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the student's latest message as a single standalone search query, "
            "resolving pronouns and references using the chat history. "
            "Return ONLY the rewritten query, nothing else.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def make_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=EMBED_MODEL)


def file_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _splitter(size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )


def load_pdf(data: bytes, filename: str, subject: str) -> list[Document]:
    """Parse a PDF into page-level Documents tagged with subject/source/page."""
    if len(data) > MAX_PDF_MB * 1024 * 1024:
        raise ValueError(f"{filename}: larger than {MAX_PDF_MB} MB.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        pages = PyPDFLoader(path).load()
    finally:
        Path(path).unlink(missing_ok=True)

    if len(pages) > MAX_PAGES:
        raise ValueError(f"{filename}: {len(pages)} pages exceeds the {MAX_PAGES}-page limit.")

    pages = [p for p in pages if p.page_content.strip()]
    if not pages:
        raise ValueError(f"{filename}: no extractable text (scanned PDF? OCR is not supported).")

    for p in pages:
        p.metadata = {
            "source": filename,
            "page": int(p.metadata.get("page", 0)) + 1,
            "subject": subject,
        }
    return pages


def add_to_store(
    store: Optional[FAISS],
    docs: Sequence[Document],
    embeddings: FastEmbedEmbeddings,
    progress: Optional[Callable[[float], None]] = None,
    splitter: Optional[RecursiveCharacterTextSplitter] = None,
) -> tuple[FAISS, int]:
    """Chunk + embed in small batches and add to (or create) a FAISS store."""
    chunks = (splitter or _splitter()).split_documents(list(docs))
    chunks = [c for c in chunks if len(c.page_content.strip()) > 30]
    if not chunks:
        raise ValueError("Nothing to index after chunking.")

    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        if store is None:
            store = FAISS.from_documents(batch, embeddings)
        else:
            store.add_documents(batch)
        if progress:
            progress(min(i + EMBED_BATCH, len(chunks)) / len(chunks))
    return store, len(chunks)


def load_default_store(embeddings: FastEmbedEmbeddings) -> FAISS:
    """Index the bundled Computer Science handbook (used for instant demos)."""
    docs = TextLoader(str(DEFAULT_CONTENT_PATH), encoding="utf-8").load()
    for d in docs:
        d.metadata = {"source": DEFAULT_SOURCE, "page": None, "subject": DEFAULT_SUBJECT}
    store, _ = add_to_store(None, docs, embeddings, splitter=_splitter(700, 100))
    return store


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def retrieve(
    query: str,
    stores: Sequence[Optional[FAISS]],
    subjects: Optional[Sequence[str]] = None,
    k: int = TOP_K,
) -> list[Document]:
    """Search every store, merge by distance (lower = closer), return top-k."""
    hits: list[tuple[Document, float]] = []
    for store in stores:
        if store is None:
            continue
        kwargs = {"k": k, "fetch_k": max(k * 10, 50)}
        if subjects:
            kwargs["filter"] = {"subject": list(subjects)}
        hits.extend(store.similarity_search_with_score(query, **kwargs))
    hits.sort(key=lambda h: h[1])
    return [d for d, _ in hits[:k]]


def format_context(docs: Sequence[Document]) -> str:
    if not docs:
        return "(no relevant context found)"
    parts = []
    for i, d in enumerate(docs, 1):
        page = d.metadata.get("page")
        loc = f", p.{page}" if page else ""
        parts.append(f"[S{i}] ({d.metadata.get('source')}{loc})\n{d.page_content.strip()}")
    return "\n\n".join(parts)


def format_sources(docs: Sequence[Document]) -> list[dict]:
    out = []
    for i, d in enumerate(docs, 1):
        snippet = " ".join(d.page_content.split())
        out.append(
            {
                "tag": f"S{i}",
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "subject": d.metadata.get("subject"),
                "snippet": snippet[:260] + ("…" if len(snippet) > 260 else ""),
            }
        )
    return out


def history_to_messages(messages: Sequence[dict], turns: int = HISTORY_TURNS) -> list[BaseMessage]:
    """Convert UI chat dicts to LangChain messages, keeping the last N turns."""
    recent = list(messages)[-turns * 2 :]
    return [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in recent
    ]


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
class Tutor:
    """Thin wrapper around Gemini with query rewriting and streaming answers."""

    def __init__(self, api_key: str, model: str):
        self.llm = ChatGoogleGenerativeAI(
            model=model, google_api_key=api_key, temperature=0.2, max_retries=2
        )
        self._rewrite = REWRITE_PROMPT | self.llm | StrOutputParser()
        self._answer = (
            ChatPromptTemplate.from_messages(
                [("system", SYSTEM_PROMPT), MessagesPlaceholder("history"), ("human", "{question}")]
            )
            | self.llm
            | StrOutputParser()
        )

    def rewrite(self, question: str, history: Sequence[BaseMessage]) -> str:
        """Make follow-ups self-contained for retrieval. Skipped on the first turn."""
        if not history:
            return question
        try:
            rewritten = self._rewrite.invoke({"question": question, "history": list(history)}).strip()
            return rewritten or question
        except Exception:  # retrieval should never fail because of the rewrite step
            return question

    def stream(
        self,
        question: str,
        history: Sequence[BaseMessage],
        docs: Sequence[Document],
        profile: StudentProfile,
        mode: str = "Q&A",
    ) -> Iterator[str]:
        yield from self._answer.stream(
            {
                "name": profile.name or "Student",
                "level": profile.level,
                "style": STYLE_GUIDE.get(profile.style, ""),
                "length": LENGTH_GUIDE.get(profile.length, ""),
                "language": profile.language,
                "goal": profile.goal or "General understanding",
                "examples": "include a short worked example" if profile.examples else "not needed",
                "mode_instruction": MODE_GUIDE.get(mode, MODE_GUIDE["Q&A"]),
                "context": format_context(docs),
                "history": list(history),
                "question": question,
            }
        )
