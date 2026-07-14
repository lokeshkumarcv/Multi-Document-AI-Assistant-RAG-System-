import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# ==========================================
# Paths
# ==========================================

DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"

# ==========================================
# Step 1: Load PDF Files
# ==========================================

def load_pdf_files(data_path):
    loader = DirectoryLoader(
        path=data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )

    documents = loader.load()
    return documents


# ==========================================
# Step 2: Split Documents into Chunks
# ==========================================

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)
    return chunks


# ==========================================
# Step 3: Load Embedding Model
# ==========================================

def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


# ==========================================
# Step 4: Create FAISS Vector Store
# ==========================================

def create_vector_store():

    print("Loading PDF files...")
    documents = load_pdf_files(DATA_PATH)
    print(f"Loaded {len(documents)} pages.")

    print("Creating text chunks...")
    chunks = create_chunks(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model...")
    embedding_model = get_embedding_model()

    print("Creating FAISS vector database...")
    db = FAISS.from_documents(chunks, embedding_model)

    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    db.save_local(DB_FAISS_PATH)

    print("\nVector database created successfully!")
    print(f"Saved at: {DB_FAISS_PATH}")


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    create_vector_store()