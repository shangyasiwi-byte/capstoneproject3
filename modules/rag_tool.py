from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import Qdrant
from langchain.schema import Document
from langchain.chains import RetrievalQA
import streamlit as st

# Load secret keys
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Inisialisasi embedding dan koneksi Qdrant
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
vectorstore = Qdrant.from_existing_collection(
    embedding=embeddings,
    collection_name="imdb_movies",
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

def retrieve_info(query, k=5):
    """Mengambil top-k hasil pencarian dari Qdrant berdasarkan query."""
    results = vectorstore.similarity_search(query, k=k)
    return [r.page_content for r in results]

# --- Buat retriever ---
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# buat chain RAG sederhana
rag_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY),
    chain_type="stuff",
    retriever=retriever,
)