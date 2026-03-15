# Florida CPI Procedure Assistant

A fully local, privacy-preserving RAG (Retrieval-Augmented Generation) application that allows Florida Child Protective Investigators (CPIs) to query the **CFOP 170-5** operating procedures manual using natural language.

**Zero-Cost. Zero-Leakage. No API keys. No data leaves your machine.**

---

## What it Does

1. Downloads the official CFOP 170-5 PDF from the Florida DCF website.
2. Chunks and embeds the document using a local HuggingFace model.
3. Stores the vectors in a local ChromaDB database.
4. Answers natural language questions by retrieving the most relevant procedural sections and passing them to a locally running LLM (Ollama / llama3.2).
5. Cites exact pages and paragraphs. Refuses to answer if context is insufficient.

---

## Disclaimer

> This is an independent portfolio project for educational and reference purposes only. It is **NOT** an official DCF application. Always consult the official CFOP 170-5 manual and your supervisor for authoritative guidance.

---

## Tech Stack

| Component        | Library / Tool                        | Why                                      |
|------------------|---------------------------------------|------------------------------------------|
| LLM              | Ollama (`llama3.2`)                   | 100% local, free, no API key             |
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2` | CPU-friendly, 80MB, no API key        |
| Vector Store     | ChromaDB (persistent local)           | Simple, file-based, no server needed     |
| Orchestration    | LangChain                             | Clean RAG chain composition              |
| PDF Parsing      | `pypdf` via LangChain loader          | Reliable, lightweight                    |
| Frontend         | Streamlit                             | Fast chat UI with source citation display|

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed
- ~2 GB disk space for models and vector store
- 8 GB RAM recommended (4 GB minimum with `llama3.2:3b`)

---

## Setup

### 1. Install Ollama and pull the model

```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3.2
```

### 2. Clone the repo and set up Python

```bash
git clone <your-repo-url>
cd "RAG 1"

python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Ingest the CFOP 170-5 document

```bash
python ingest.py
```

This will:
- Download the CFOP 170-5 PDF (~5 MB) into `data/`
- Download the embedding model (~80 MB, cached after first run)
- Build the ChromaDB vector store in `florida_db/`

Takes ~2–5 minutes on first run. Subsequent runs skip the download if files exist.

### 4. Start the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Usage

Ask questions about Florida CPI procedures in plain English:

- *"What is the required response time for a Priority 1 investigation?"*
- *"What are the criteria for a child to be classified as a victim?"*
- *"What documentation is required within 24 hours of initiating an investigation?"*

The assistant will:
- Answer using only text found in the CFOP 170-5 manual
- Cite the page and paragraph it sourced from
- Show the raw retrieved chunks in an expandable panel
- Say **"I cannot find this in the loaded CFOP manual"** if the answer is not present

---

## Project Structure

```
RAG 1/
├── .gitignore          # Excludes PDFs, vector DB, venv, caches
├── README.md
├── requirements.txt
├── ingest.py           # Downloads PDF, chunks, embeds, saves to ChromaDB
├── app.py              # Streamlit chat UI + RAG chain
├── data/
│   └── .gitkeep        # Folder tracked; actual PDF is gitignored
└── florida_db/
    └── .gitkeep        # Folder tracked; vector DB files are gitignored
```

---

## Architecture

```
INGEST:
  DCF Website (PDF) ──► pypdf loader ──► RecursiveCharacterTextSplitter
                    ──► HuggingFace Embeddings ──► ChromaDB (local)

QUERY:
  User Question ──► HuggingFace Embeddings ──► ChromaDB similarity search
               ──► Top 5 chunks ──► Prompt template ──► Ollama (llama3.2)
               ──► Streamed answer + cited sources ──► Streamlit UI
```

---

## Model Options

Switch the model in `app.py` by changing `OLLAMA_MODEL`:

| Model            | RAM Required | Speed  | Quality  |
|------------------|-------------|--------|----------|
| `llama3.2`       | 4–8 GB      | Medium | Good     |
| `llama3.2:3b`    | 4 GB        | Fast   | Moderate |
| `mistral`        | 4–8 GB      | Medium | Good     |
| `phi3:mini`      | 2–4 GB      | Fast   | Moderate |

---

## Privacy & Security

- No API keys anywhere in this codebase.
- No calls to OpenAI, Anthropic, Cohere, or any external service.
- The only outbound network calls are:
  - One-time PDF download from `myflfamilies.com` (official DCF domain)
  - One-time HuggingFace model download (~80 MB, cached locally)
- All inference runs on `localhost` via Ollama.
- `.gitignore` excludes the PDF and vector store to prevent accidental data commits.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Connection refused on port 11434` | Ollama is not running. Run `ollama serve` |
| `Vector store not found` | Run `python ingest.py` first |
| PDF download fails | Download manually from [myflfamilies.com](https://www.myflfamilies.com) and place at `data/CFOP_170-5.pdf` |
| Slow on first run | Normal — embedding model is downloading (~80 MB). Cached after that |
| Out of memory | Switch to `phi3:mini` in `app.py` (`OLLAMA_MODEL = "phi3:mini"`) |
