import os
import re
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def pdf_to_bangla_text(pdf_path: str) -> str:
    print("[*] Converting PDF pages to images...")
    images = convert_from_path(pdf_path)

    print("[*] Running Bangla OCR on each page...")
    full_text = ""
    for i, img in enumerate(images):
        img_cv = np.array(img)
        img_cv = img_cv[:, :, ::-1].copy()  # Convert RGB to BGR
        text = pytesseract.image_to_string(img_cv, lang='ben')
        full_text += text + "\n"
        print(f"[+] OCR done for page {i + 1}")
    
    return clean_text(full_text)

def chunk_and_vectorize(text: str, persist_directory: str = "chroma_db"):
    print("[*] Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "।", "!", "?", " "]
    )
    documents = splitter.create_documents([text])

    print("[*] Generating embeddings and storing in ChromaDB...")
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print("[✅] Vector store created and saved at:", persist_directory)

if __name__ == "__main__":
    pdf_path = "data/HSC26-Bangla1st-Paper.pdf"
    print(f"[🚀] Processing PDF: {pdf_path}")
    
    bangla_text = pdf_to_bangla_text(pdf_path)
    chunk_and_vectorize(bangla_text)
