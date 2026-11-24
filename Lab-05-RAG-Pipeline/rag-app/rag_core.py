"""Core RAG logic shared by ingest.py, rag_pipeline.py and app.py."""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterator

import pymupdf  # PyMuPDF
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ----------------------------------------------------------------- settings
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION = "university_docs"
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5
# Optional: drop chunks whose cosine distance is above this value (0 = identical,
# ~1 = unrelated). None = keep everything and let the prompt decide. Tune it
# after you look at the distances shown in the app.
MAX_DISTANCE: float | None = None

NOT_FOUND = "I couldn't find this in the uploaded documents."

# ------------------------------------------------------------------ prompts
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a university knowledge assistant. Answer the question using "
            "ONLY the context provided.\n"
            "Rules:\n"
            "1. If the answer is not in the context, reply exactly: "
            f"\"{NOT_FOUND}\" and nothing else.\n"
            "2. Never use outside knowledge and never guess.\n"
            "3. Be concise. Use a numbered or bulleted list when listing items "
            "(e.g. subjects, requirements) and keep module codes exactly as written.\n"
            "4. Cite the source after each key fact like this: (file.pdf, p.3).\n"
            "5. If the context contains conflicting information, point it out.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    "Given the chat history and a follow-up question, rewrite the follow-up as a "
    "single standalone question that can be understood without the history. "
    "Do NOT answer it. If it is already standalone, return it unchanged. "
    "Return only the question.\n\n"
    "Chat history:\n{history}\n\nFollow-up question: {question}\n\nStandalone question:"
)


# ------------------------------------------------------------- health check
def check_ollama() -> tuple[bool, str]:
    """Verify Ollama is running and both models are pulled."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            names = {m["name"] for m in json.load(r)["models"]}
    except Exception:
        return False, (
            f"Cannot reach Ollama at {OLLAMA_URL}. Start the Ollama app "
            "(or run `ollama serve`) and try again."
        )
    missing = [
        m for m in (LLM_MODEL, EMBED_MODEL)
        if m not in names and f"{m}:latest" not in names
    ]
    if missing:
        return False, "Missing models. Run: " + " ; ".join(f"ollama pull {m}" for m in missing)
    return True, "ok"


# --------------------------------------------------------------- vector DB
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL),
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},  # distances in [0, 2]
    )


def list_documents(vs: Chroma) -> dict[str, int]:
    """Return {filename: number_of_chunks}."""
    metas = vs.get(include=["metadatas"])["metadatas"]
    return dict(Counter(m["source"] for m in metas))


def delete_document(vs: Chroma, source: str) -> None:
    ids = vs.get(where={"source": source})["ids"]
    if ids:
        vs.delete(ids=ids)


# --------------------------------------------------------------- ingestion
def pdf_to_documents(data: bytes, filename: str, file_hash: str) -> list[Document]:
    """One Document per non-empty page, tagged with source + page number."""
    docs = []
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        for i, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": filename, "page": i, "file_hash": file_hash},
                    )
                )
    return docs


def ingest_pdf(vs: Chroma, data: bytes, filename: str) -> int:
    """Add a PDF to the knowledge base.

    Returns the number of chunks added, or 0 if the identical file is already
    indexed. If a file with the same name but different content exists, the old
    version is replaced.
    """
    file_hash = hashlib.sha256(data).hexdigest()[:16]

    existing = vs.get(where={"source": filename}, limit=1, include=["metadatas"])
    if existing["ids"]:
        if existing["metadatas"][0].get("file_hash") == file_hash:
            return 0
        delete_document(vs, filename)  # updated version of the same file

    pages = pdf_to_documents(data, filename, file_hash)
    if not pages:
        raise ValueError("No extractable text (scanned PDF? it would need OCR).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    ids = [f"{file_hash}-{i}" for i in range(len(chunks))]
    vs.add_documents(chunks, ids=ids)
    return len(chunks)


def ingest_folder(vs: Chroma, folder: Path = DATA_DIR) -> dict[str, str]:
    """Ingest every PDF in a folder. Returns {filename: status message}."""
    folder.mkdir(exist_ok=True)
    report = {}
    for pdf in sorted(folder.glob("*.pdf")):
        try:
            n = ingest_pdf(vs, pdf.read_bytes(), pdf.name)
            report[pdf.name] = f"{n} chunks added" if n else "already indexed"
        except Exception as e:  # keep going if one file is bad
            report[pdf.name] = f"FAILED: {e}"
    return report


# --------------------------------------------------------------- retrieval
def retrieve(
    vs: Chroma, question: str, sources: list[str] | None = None, k: int = TOP_K
) -> list[tuple[Document, float]]:
    if vs._collection.count() == 0:
        return []
    flt = {"source": {"$in": list(sources)}} if sources else None
    results = vs.similarity_search_with_score(question, k=k, filter=flt)
    if MAX_DISTANCE is not None:
        results = [(d, s) for d, s in results if s <= MAX_DISTANCE]
    return results


def format_context(results: list[tuple[Document, float]]) -> str:
    return "\n\n".join(
        f"[Source: {d.metadata['source']}, p.{d.metadata['page']}]\n{d.page_content}"
        for d, _ in results
    )


def source_cards(results: list[tuple[Document, float]]) -> list[dict]:
    return [
        {
            "label": f"{d.metadata['source']} - p.{d.metadata['page']}",
            "distance": round(float(s), 3),
            "snippet": d.page_content[:300].replace("\n", " ") + "...",
        }
        for d, s in results
    ]


# -------------------------------------------------------------- generation
def _llm() -> ChatOllama:
    # num_ctx=4096: Ollama's default context can silently truncate long prompts.
    return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0, num_ctx=4096)


def condense_question(question: str, history: list[dict]) -> str:
    """Turn a follow-up like 'what about year 2?' into a standalone question."""
    if not history:
        return question
    hist = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:])
    chain = CONDENSE_PROMPT | _llm() | StrOutputParser()
    return chain.invoke({"history": hist, "question": question}).strip() or question


def answer_stream(question: str, results: list[tuple[Document, float]]) -> Iterator[str]:
    if not results:
        yield NOT_FOUND
        return
    chain = ANSWER_PROMPT | _llm() | StrOutputParser()
    yield from chain.stream({"context": format_context(results), "question": question})


def ask(vs, question, history=None, sources=None, k=TOP_K):
    """Convenience wrapper: returns (standalone_question, results)."""
    standalone = condense_question(question, history or [])
    return standalone, retrieve(vs, standalone, sources, k)
