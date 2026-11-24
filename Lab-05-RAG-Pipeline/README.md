# Lab 05: RAG Pipeline (University AI Assistant)

## Objective
Build a Retrieval-Augmented Generation (RAG) application that answers questions from university PDF documents, and extend it so users can index multiple PDFs and upload their own.

## Architecture
Document Loader (PyMuPDF) -> Text Splitter -> Embeddings -> Vector Store (ChromaDB) -> Retriever -> Prompt Template -> LLM -> Answer with sources

| Component | Tool |
|-----------|------|
| LLM | Llama 3.2 3B via Ollama |
| Embedding model | Qwen3 Embedding 0.6B via Ollama |
| Vector database | ChromaDB (cosine similarity) |
| Framework | LangChain |
| Chat interface | Streamlit (web) and terminal |

Settings: chunk size 1000 characters, overlap 150, top-k = 5 retrieved passages, temperature 0.

## Features
- Indexes every PDF in the `data/` folder (`ingest.py`)
- Terminal chat (`rag_pipeline.py`) and web chat (`app.py`)
- Users can upload multiple PDFs in the Streamlit sidebar and ask questions straight away
- Duplicate files are skipped, and updated versions of a file replace the old one
- Documents can be removed, and questions can be limited to selected documents
- Follow-up questions are rewritten into standalone questions using the chat history
- Answers cite the source file and page number
- If the answer is not in the documents, the assistant says so instead of guessing

## Project Files (`rag-app/`)
| File | Purpose |
|------|---------|
| `rag_core.py` | Shared RAG logic: ingestion, retrieval, prompts, generation |
| `ingest.py` | Indexes PDFs from `data/` into ChromaDB |
| `rag_pipeline.py` | Terminal chat |
| `app.py` | Streamlit chat app with PDF upload |
| `requirements.txt` | Python dependencies |

## How to Run
```powershell
py -3.12 -m venv rag_env
.\rag_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

ollama pull llama3.2:3b
ollama pull qwen3-embedding:0.6b

mkdir data                  # put your PDFs here
python ingest.py            # index the PDFs (add --reset to start fresh)
python rag_pipeline.py      # terminal chat
streamlit run app.py        # web chat with PDF upload
```
Ollama must be running before starting the app.

## Screenshots
### Models installed
![ollama list](./screenshots/01-ollama-list.png)
### Ingestion
![ingest](./screenshots/02-ingest.png)
### Terminal chat
![terminal chat](./screenshots/03-terminal-chat.png)
### Streamlit app
![streamlit answer](./screenshots/04-streamlit-answer.png)
### Uploading a PDF
![upload](./screenshots/05-upload-pdf.png)
![answer from upload](./screenshots/06-answer-from-upload.png)
### Question not in the documents
![not found](./screenshots/07-not-found.png)

## Key Learnings
- RAG grounds the LLM in your own documents, which reduces made-up answers.
- Chunk size, overlap and top-k affect answer quality.
- A strict prompt ("use only the context") makes the model refuse when the answer is missing.
- Keeping page numbers as metadata allows source citations.

## Limitations
- Scanned PDFs without text need OCR and are not supported.
- A small 3B model can miss details when information is spread across many chunks.
- Fixed-size chunking can split related information. Advanced approaches such as hybrid retrieval and reranking could improve this.