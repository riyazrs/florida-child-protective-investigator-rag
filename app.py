"""
app.py — Streamlit chat interface for the Florida CPI Assistant.

Start the app:
    streamlit run app.py

Prerequisites:
    1. Run `python ingest.py` first to build the vector store.
    2. Ollama must be running: `ollama serve` (starts automatically on most installs).
    3. Pull the model once: `ollama pull llama3.2`
"""

import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VECTOR_DB_DIR = "./florida_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"
TOP_K_RESULTS = 5

SYSTEM_PROMPT_TEMPLATE = """You are a strict legal reference assistant for Florida Child Protective Investigators (CPIs).
Your ONLY job is to answer questions based on the CFOP 170-5 operating procedures manual excerpts provided below.

RULES YOU MUST FOLLOW:
1. Answer ONLY using information found in the context below. Do not use any outside knowledge.
2. If the answer is not present in the context, respond EXACTLY with: "I cannot find this in the loaded CFOP manual."
3. Always cite the specific Chapter, Paragraph, or Section number from the retrieved text when available.
4. Be concise and factual. This is a legal/procedural reference tool, not a conversational assistant.
5. Never speculate, infer, or fill gaps with assumed knowledge.

---
CONTEXT FROM CFOP 170-5:
{context}
---

QUESTION: {question}

ANSWER (cite specific paragraphs where possible):"""


# ---------------------------------------------------------------------------
# Cached resource loading (runs once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def load_retriever():
    if not os.path.exists(VECTOR_DB_DIR) or not os.listdir(VECTOR_DB_DIR):
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
        collection_name="cfop_170_5",
    )
    return db.as_retriever(search_kwargs={"k": TOP_K_RESULTS})


@st.cache_resource(show_spinner="Connecting to Ollama...")
def load_llm():
    return Ollama(model=OLLAMA_MODEL, temperature=0)


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Florida CPI Assistant",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Florida CPI Procedure Assistant")
st.caption("Powered by CFOP 170-5 · Runs 100% locally · No data leaves your machine")

st.warning(
    "**DISCLAIMER:** This is an independent portfolio project for educational and "
    "reference purposes only. It is NOT an official DCF application. Always consult "
    "the official CFOP 170-5 manual and your supervisor for authoritative guidance.",
    icon="⚠️",
)

st.divider()

# ---------------------------------------------------------------------------
# Check prerequisites
# ---------------------------------------------------------------------------

retriever = load_retriever()

if retriever is None:
    st.error(
        "Vector store not found. Please run the ingestion script first:\n\n"
        "```\npython ingest.py\n```"
    )
    st.stop()

try:
    llm = load_llm()
except Exception:
    st.error(
        "Could not connect to Ollama. Make sure it is running:\n\n"
        "```\nollama serve\n```\n\n"
        "And that the model is pulled:\n\n"
        "```\nollama pull llama3.2\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Build the RAG chain
# ---------------------------------------------------------------------------

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=SYSTEM_PROMPT_TEMPLATE,
)


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📄 Source Documents Used"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Chunk {i}** — Page {source.metadata.get('page', 'N/A')}")
                    st.text(source.page_content[:400] + "...")
                    st.divider()

if user_question := st.chat_input("Ask a question about CFOP 170-5 procedures..."):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        # Retrieve source docs for display
        source_docs = retriever.invoke(user_question)

        # Stream the LLM response
        response_placeholder = st.empty()
        full_response = ""

        with st.spinner("Searching CFOP manual..."):
            for chunk in rag_chain.stream(user_question):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

        # Show source documents used
        with st.expander("📄 Source Documents Used"):
            for i, doc in enumerate(source_docs, 1):
                st.markdown(f"**Chunk {i}** — Page {doc.metadata.get('page', 'N/A')}")
                st.text(doc.page_content[:400] + "...")
                st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": source_docs,
    })
