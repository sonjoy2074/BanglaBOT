import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import re
from dotenv import load_dotenv

load_dotenv()

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_and_preprocess(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    docs = [clean_text(doc.page_content) for doc in pages]
    return docs

def chunk_and_vectorize(docs, persist_directory="chroma_db"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "।", "!", "?", " "]
    )
    chunks = splitter.create_documents(docs)

    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb

if __name__ == "__main__":
    path = "data/HSC26-Bangla1st-Paper.pdf"
    docs = load_and_preprocess(path)
    chunk_and_vectorize(docs)
