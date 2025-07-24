import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os

# === LOAD ENV ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === CONFIGURATION ===
CHROMA_DB_DIR = "bangla_chroma_db"
COLLECTION_NAME = "bangla_book"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OPENAI_MODEL = "gpt-3.5-turbo"  # or "gpt-4"

# === LOAD CHROMADB COLLECTION ===
@st.cache_resource
def load_chroma_collection():
    client_chroma = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return client_chroma.get_or_create_collection(COLLECTION_NAME)

# === LOAD EMBEDDING MODEL ===
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)

collection = load_chroma_collection()
embedder = load_embedding_model()

# === INIT MEMORY ===
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based only on the given Bangla book context."}
    ]

# === UI ===
st.title("🧠 BanglaBot: Bangla & English Study Assistant")

user_query = st.text_input("🔎 Enter your question (Bangla or English):")

if user_query:
    st.info("🔍 Searching relevant chunks...")

    query_embedding = embedder.encode([user_query]).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    if results['documents'] and results['documents'][0]:
        context = "\n".join(results['documents'][0])
    else:
        context = "No relevant information found."

    st.success("📚 Context fetched. Generating answer...")

    full_prompt = f"Context:\n{context}\n\nQuestion: {user_query}"
    st.session_state.messages.append({"role": "user", "content": full_prompt})

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=st.session_state.messages
    )

    answer = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": answer})

    st.markdown("### 🤖 Answer:")
    st.write(answer)

# === Conversation History Display ===
with st.expander("🧾 Conversation History"):
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            st.markdown(f"**{msg['role'].capitalize()}**: {msg['content']}")
