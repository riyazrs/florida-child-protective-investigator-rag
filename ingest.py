"""
ingest.py — Downloads CFOP 170-5 and builds the local ChromaDB vector store.

Run once before starting the app:
    python ingest.py

No API keys required. Everything runs locally.
"""

import os
import sys
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PDF_URL = (
    "https://www.myflfamilies.com/sites/default/files/2023-01/CFOP%20170-5.pdf"
)
PDF_PATH = os.path.join("data", "CFOP_170-5.pdf")
VECTOR_DB_DIR = "./florida_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Larger chunks preserve legal/procedural context across paragraph boundaries.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


# ---------------------------------------------------------------------------
# Step 1 — Download the source PDF
# ---------------------------------------------------------------------------

def download_pdf(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"[ingest] PDF already exists at '{dest}', skipping download.")
        return

    print(f"[ingest] Downloading CFOP 170-5 from DCF website...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    try:
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"[ingest] ERROR: Could not download PDF — {e}")
        print(
            "[ingest] Please download CFOP 170-5 manually from "
            "https://www.myflfamilies.com and place it at: data/CFOP_170-5.pdf"
        )
        sys.exit(1)

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"[ingest] PDF saved to '{dest}'.")


# ---------------------------------------------------------------------------
# Step 2 — Load and chunk the PDF
# ---------------------------------------------------------------------------

def load_and_split(pdf_path: str) -> list:
    print(f"[ingest] Loading PDF and splitting into chunks...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"[ingest] Loaded {len(pages)} pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    print(f"[ingest] Created {len(chunks)} text chunks.")
    return chunks


# ---------------------------------------------------------------------------
# Step 3 — Embed and persist to ChromaDB
# ---------------------------------------------------------------------------

def build_vector_store(chunks: list) -> None:
    print(f"[ingest] Loading embedding model '{EMBEDDING_MODEL}'...")
    print("[ingest] (First run will download ~80MB model — cached after that)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"[ingest] Embedding {len(chunks)} chunks and saving to ChromaDB...")
    print("[ingest] This may take a few minutes on first run.")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
        collection_name="cfop_170_5",
    )

    print(f"[ingest] Vector store saved to '{VECTOR_DB_DIR}'.")
    print("[ingest] Ingestion complete. You can now run: streamlit run app.py")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    download_pdf(PDF_URL, PDF_PATH)
    chunks = load_and_split(PDF_PATH)
    build_vector_store(chunks)
