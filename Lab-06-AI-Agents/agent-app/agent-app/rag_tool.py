import os
import sys
import pickle
import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool

DATA_DIR = "data"              # put handbook.pdf etc. here
INDEX_FILE = "rag_index.pkl"
TOP_K = 4

embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
_index = None                  # loaded lazily


def _normalize(m):
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def build_index():
    """Load PDFs -> split into chunks -> embed -> save to disk."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = []
    for fn in sorted(os.listdir(DATA_DIR)):
        if not fn.lower().endswith(".pdf"):
            continue
        reader = PdfReader(os.path.join(DATA_DIR, fn))
        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for piece in splitter.split_text(text):
                chunks.append({"text": piece, "source": fn, "page": page_no})

    if not chunks:
        sys.exit(f"No text found. Put PDFs in ./{DATA_DIR}")

    print(f"Embedding {len(chunks)} chunks (this can take a minute)...")
    vectors = np.array(embeddings.embed_documents([c["text"] for c in chunks]))
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"chunks": chunks, "vectors": _normalize(vectors)}, f)
    print(f"Saved index to {INDEX_FILE}")


@tool
def search_university_knowledge(query: str) -> str:
    """Search the university knowledge base (handbook, programme details,
    courses, modules, policies, regulations). Use this for ANY question about
    the university. Pass the user's question as the query."""
    global _index
    if _index is None:
        if not os.path.exists(INDEX_FILE):
            return "Knowledge base not built. Run: python rag_tool.py build"
        with open(INDEX_FILE, "rb") as f:
            _index = pickle.load(f)

    q = _normalize(np.array(embeddings.embed_query(query)))
    scores = _index["vectors"] @ q                     # cosine similarity
    best = np.argsort(scores)[::-1][:TOP_K]

    results = []
    for i in best:
        c = _index["chunks"][i]
        results.append(f"Source: {c['source']}, Page: {c['page']}\n{c['text']}")
    return "\n\n---\n\n".join(results)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_index()
    elif len(sys.argv) > 2 and sys.argv[1] == "test":
        print(search_university_knowledge.invoke({"query": " ".join(sys.argv[2:])}))
    else:
        print("Usage:\n  python rag_tool.py build\n  python rag_tool.py test <question>")
