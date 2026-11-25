```powershell
mkdir rag_test; cd rag_test            # copy these files in here
py -3.12 -m venv rag_env
# if activation is blocked: Set-ExecutionPolicy -Scope Process Bypass
.\rag_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

ollama pull llama3.2:3b
ollama pull qwen3-embedding:0.6b
ollama list                             # both models should appear

mkdir data                              # put your PDFs here
python ingest.py                        # index them (use --reset to start fresh)
python rag_pipeline.py                  # terminal chat
streamlit run app.py                    # web chat with PDF upload
```

Optional `.env` overrides: LLM_MODEL, EMBED_MODEL, OLLAMA_URL, CHROMA_DIR, DATA_DIR
