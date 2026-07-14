# 🧠 Multi-Document AI Assistant (RAG System)

## 1. Overview

This project implements an end-to-end RAG pipeline:

1. PDF documents are loaded, chunked, and embedded into a local vector database (FAISS).
2. At query time, the most relevant chunks are retrieved from the vector database.
3. A large language model (via Groq) uses the retrieved chunks as context to generate a grounded answer.
4. A Streamlit web app provides a chat interface for interacting with the assistant.

---

## 2. Architecture

```
                     ┌─────────────────────────┐
                     │   PDF Documents (data/)  │
                     └────────────┬─────────────┘
                                  │
                     create_memory_for_llm.py
                                  │
              ┌───────────────────────────────────┐
              │ 1. Load PDFs (DirectoryLoader)     │
              │ 2. Split into chunks (RecursiveCTS)│
              │ 3. Embed chunks (MiniLM-L6-v2)     │
              │ 4. Store vectors in FAISS index    │
              └────────────────┬────────────────────┘
                                │
                     vectorstore/db_faiss
                                │
        ┌───────────────────────┴───────────────────────┐
        │                                                │
connect_memory_with_llm.py                          app.py (Streamlit)
   (CLI Q&A loop)                                (Web chat interface)
        │                                                │
        └───────────────── create_retrieval_chain ───────┘
                                │
                     ┌───────────────────┐
                     │  Groq LLM          │
                     │ (llama-3.1-8b-     │
                     │  instant)          │
                     └───────────────────┘
```

---

## 3. Tech Stack

| Component | Technology |
|---|---|
| LLM Provider | [Groq](https://groq.com/) (`llama-3.1-8b-instant`) |
| Orchestration Framework | LangChain (`langchain-classic`, `langchain-core`, `langchain-community`) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (local, file-based) |
| PDF Parsing | `pypdf` via `PyPDFLoader` |
| Web Interface | Streamlit |
| Config Management | `python-dotenv` |

---

## 4. Project Structure

```
project-root/
│
├── data/                          # Source PDF documents (input corpus)
├── vectorstore/
│   └── db_faiss/                  # Persisted FAISS vector index (generated)
│
├── create_memory_for_llm.py       # Step 1: Build the vector database from PDFs
├── connect_memory_with_llm.py     # Step 2: CLI tool to query the RAG chain
├── app.py                         # Step 3: Streamlit web app for chat-based Q&A
├── requirements.txt                # Python dependencies
└── .env                            # Environment variables (GROQ_API_KEY)
```

---

## 5. Setup & Installation

### 5.1 Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com/)

### 5.2 Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The current `requirements.txt` contains duplicate entries (e.g. `langchain`, `torch`, `faiss-cpu` are listed twice) and an invalid package name (`dotenv` — the correct package is `python-dotenv`). It is recommended to clean this file to avoid installation conflicts. A de-duplicated version should include:
>
> ```
> langchain
> langchain-classic
> langchain-community
> langchain-core
> langchain-groq
> langchain-huggingface
> langchain-text-splitters
> faiss-cpu
> sentence-transformers
> huggingface-hub
> transformers
> torch
> pypdf
> python-dotenv
> streamlit
> torchvision
> ```


### 5.3 Configure environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_Hugging_face_token
```

---

## 6. Usage

The pipeline is used in three sequential steps.

### Step 1 — Build the Vector Database

Place your PDF files in the `data/` folder, then run:

```bash
python create_memory_for_llm.py
```

**What it does:**
- Loads all `.pdf` files from `data/` using `DirectoryLoader` + `PyPDFLoader`.
- Splits documents into chunks of **1000 characters** with **100-character overlap** using `RecursiveCharacterTextSplitter`.
- Generates embeddings for each chunk using the `all-MiniLM-L6-v2` sentence-transformer model.
- Builds a FAISS vector index from the embedded chunks.
- Saves the index locally to `vectorstore/db_faiss`.

### Step 2 — Query via Command Line (optional)

```bash
python connect_memory_with_llm.py
```

**What it does:**
- Loads the persisted FAISS index and embedding model.
- Initializes the Groq LLM (`llama-3.1-8b-instant`, temperature `0.3`).
- Builds a retrieval chain that fetches the top **3** most relevant chunks (`k=3`) per query.
- Runs an interactive loop in the terminal — type a question, get an answer plus the source document chunks used, and type `exit` to quit.

### Step 3 — Run the Web Application

```bash
streamlit run app.py
```

**What it does:**
- Loads the same FAISS index and Groq LLM (temperature `0.3`, `max_tokens=512`), cached via `@st.cache_resource` so the resources load only once per session.
- Presents a styled chat interface ("🧠 Multi-Document AI Assistant") where users can type questions.
- Maintains conversation history in `st.session_state`.
- Displays the assistant's answer, grounded in the retrieved document context.
- Displays an error message in the UI if the chain invocation fails.

---

## 7. How the RAG Chain Works

Both `connect_memory_with_llm.py` and `app.py` build the same core chain:

1. **Retriever** — `db.as_retriever(search_kwargs={"k": 3})` fetches the 3 most similar chunks to the user's query from FAISS.
2. **Prompt Template** — A system prompt instructs the LLM to answer strictly using the provided context, and to say "I don't know" if the answer isn't present.
3. **Document Chain** — `create_stuff_documents_chain` "stuffs" all retrieved chunks into the prompt context.
4. **Retrieval Chain** — `create_retrieval_chain` wires the retriever and document chain together, returning both the `answer` and the source `context` documents.

---

## 8. Configuration Reference

| Setting | Location | Value |
|---|---|---|
| Chunk size | `create_memory_for_llm.py` | 1000 characters |
| Chunk overlap | `create_memory_for_llm.py` | 100 characters |
| Embedding model | all scripts | `sentence-transformers/all-MiniLM-L6-v2` |
| LLM model | `app.py`, `connect_memory_with_llm.py` | `llama-3.1-8b-instant` (Groq) |
| Retrieval count (k) | `app.py`, `connect_memory_with_llm.py` | 3 |
| Temperature | `app.py` | 0.3 |
| Temperature | `connect_memory_with_llm.py` | 0.5 |
| Max tokens | `app.py`, `connect_memory_with_llm.py` | 512 |
| Vector store path | all scripts | `vectorstore/db_faiss` |

---

## 9. Known Limitations & Suggested Improvements

- **`allow_dangerous_deserialization=True`** is used when loading the FAISS index. This is safe only if the index file is trusted and generated locally; avoid loading FAISS indexes from untrusted sources.
- **No incremental indexing** — re-running `create_memory_for_llm.py` rebuilds the entire FAISS index from scratch rather than adding only new documents.
- **No re-ranking or chunk deduplication** — larger document sets may benefit from a re-ranker or metadata filtering on top of the base similarity search.

---

## 10. Summary

This project is a lightweight, locally-hosted RAG assistant that combines FAISS for vector search, HuggingFace sentence-transformers for embeddings, and Groq-hosted Llama 3.1 for generation, exposed through both a CLI tool and a Streamlit chat interface. It follows a clean three-stage pipeline: **index → retrieve → generate**, making it straightforward to swap in different document sets, embedding models, or LLMs as needed.