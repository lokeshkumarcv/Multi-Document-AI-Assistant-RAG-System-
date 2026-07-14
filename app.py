import os
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"


@st.cache_resource
def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        max_tokens=512,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the given context to answer the question. "
                   "If you don't know the answer, say you don't know.\n\nContext:\n{context}"),
        ("human", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return create_retrieval_chain(retriever, document_chain)


st.set_page_config(page_title="Multi-Document AI Assistant", page_icon="🧠", layout="wide")

st.markdown("""
<style>
#MainMenu, footer, header {visibility:hidden;}
.main {background:#F4F7FB;}
.title {text-align:center; font-size:46px; font-weight:800;
    background:linear-gradient(90deg,#2563EB,#9333EA);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0;}
.subtitle {text-align:center; font-size:17px; color:#6b7280; margin-bottom:28px;}
.stChatMessage {border-radius:14px; padding:10px; box-shadow:0 1px 3px rgba(0,0,0,0.06);}
section[data-testid="stSidebar"] {background:#0f172a;}
section[data-testid="stSidebar"] * {color:#e2e8f0 !important;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🧠 Multi-Document AI Assistant")
    st.markdown("---")
    st.markdown(
        "- Ingest multiple PDFs into one knowledge base\n"
        "- Ask questions across all documents\n"
        "- Answers are grounded in your source files"
    )
    st.markdown("---")
    st.success("Ready to answer your questions!")

st.markdown("<div class='title'>🧠 Multi-Document AI Assistant</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Ask questions across all your uploaded documents and get accurate, sourced answers.</div>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask anything from your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Searching documents..."):
        try:
            rag_chain = get_rag_chain()
            response = rag_chain.invoke({"input": question})
            answer = response["answer"]

            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(str(e))