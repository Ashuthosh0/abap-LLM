from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from data.data_URL import urls
import os

persist_dir = os.getenv("CHROMA_DB_PATH", "./chroma_db")


loader = WebBaseLoader(urls, bs_get_text_kwargs={"strip": True})
try:
    
    docs = loader.load()
except Exception as e:
    print(f"Error loading documents: {e}")
    docs = []


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs) if docs else []

embedding_function = HuggingFaceEmbeddings()

vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_function, persist_directory=persist_dir)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


def get_retriever():
    return retriever


