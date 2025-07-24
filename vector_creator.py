import os
import re
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# === CONFIGURATION ===
BOOK_PATH = "data/HSC26-Bangla1st-Paper.pdf"  # Your Bangla book in the data/ folder
CHROMA_DB_DIR = "bangla_chroma_db"  # ChromaDB persistence folder
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Embedding model that supports Bangla

# === STEP 1: EXTRACT TEXT FROM PDF ===
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# === STEP 2: PREPROCESS BANGLA TEXT ===
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = text.strip()
    return text

# === STEP 3: CHUNK THE TEXT ===
def chunk_text(text, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "।", ".", "!", "?"]
    )
    return splitter.split_text(text)

# === STEP 4: EMBEDDINGS + CHROMADB ===
def store_in_chromadb(chunks, embedding_model, db_path):
    # Remove old DB if exists
    if os.path.exists(db_path):
        os.system(f"rm -rf {db_path}")

    # Initialize ChromaDB with persistence
    chroma_client = chromadb.Client(Settings(persist_directory=db_path))
    collection = chroma_client.create_collection(name="bangla_book")

    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

    # Save to disk
    chroma_client.persist()

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("📖 Extracting text from PDF...")
    raw_text = extract_text_from_pdf(BOOK_PATH)

    print("🧹 Preprocessing text...")
    clean_text = preprocess_text(raw_text)

    print("✂️ Chunking text...")
    chunks = chunk_text(clean_text)

    print("🔍 Loading embedding model...")
    embed_model = SentenceTransformer(MODEL_NAME)

    print("💾 Storing data in ChromaDB...")
    store_in_chromadb(chunks, embed_model, CHROMA_DB_DIR)

    print("✅ Vector database created and saved to:", CHROMA_DB_DIR)
