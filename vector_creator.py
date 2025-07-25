import os
import re
import fitz  # PyMuPDF
from chromadb import PersistentClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

#Configuration
# Define constants for file paths and model names
BOOK_PATH = "data/HSC26-Bangla1st-Paper.pdf"
CHROMA_DB_DIR = "bangla_chroma_db"
COLLECTION_NAME = "bangla_book"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

#STEP 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

#STEP 2: Clean text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text) 
    return text.strip()

#STEP 3: Split text into chunks
def chunk_text(text, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "।", ".", "!", "?"]
    )
    return splitter.split_text(text)

#STEP 4: Store chunks in ChromaDB 
def store_in_chromadb(chunks, embedding_model, db_path, collection_name):
    # Clean previous DB
    if os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)

    client = PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)

    print(f"Adding {len(chunks)} chunks to ChromaDB...")

    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

    print("Chunks stored successfully.")

# Main execution
if __name__ == "__main__":
    print("Extracting text from PDF...")
    raw_text = extract_text_from_pdf(BOOK_PATH)

    print("Preprocessing text...")
    clean_text = preprocess_text(raw_text)

    print("Chunking text...")
    chunks = chunk_text(clean_text)

    print("Loading embedding model...")
    embed_model = SentenceTransformer(MODEL_NAME)

    print("Storing data in ChromaDB...")
    store_in_chromadb(chunks, embed_model, CHROMA_DB_DIR, COLLECTION_NAME)
