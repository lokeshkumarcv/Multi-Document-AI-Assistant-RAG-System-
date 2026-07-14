import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# ==============================
# Step 1: Setup Groq LLM
# ==============================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
    max_tokens=512,
)

# ==============================
# Step 2: Load Embedding Model
# ==============================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==============================
# Step 3: Load FAISS Database
# ==============================

DB_FAISS_PATH = "vectorstore/db_faiss"

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True,
)

# ==============================
# Step 4: Prompt
# ==============================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{input}"),
    ]
)

# ==============================
# Step 5: Create Document Chain
# ==============================

document_chain = create_stuff_documents_chain(
    llm,
    prompt,
)

# ==============================
# Step 6: Create Retriever
# ==============================

retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

# ==============================
# Step 7: Create Retrieval Chain
# ==============================

rag_chain = create_retrieval_chain(
    retriever,
    document_chain,
)

# ==============================
# Step 8: Chat Loop
# ==============================

while True:

    query = input("\nWrite Query Here (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    response = rag_chain.invoke(
        {
            "input": query
        }
    )

    print("\nRESULT:\n")
    print(response["answer"])

    print("\nSOURCE DOCUMENTS:\n")

    for doc in response["context"]:
        print(doc.metadata)
        print(doc.page_content[:200])
        print("-" * 50)