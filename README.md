# Bangla-English RAG Chatbot

This is a Retrieval-Augmented Generation (RAG) based chatbot designed to answer questions in **Bangla** and **English** using content extracted from Bangla educational PDFs (e.g., HSC textbooks). It supports:
- PDF OCR extraction using Tesseract (Bangla language)
- Embedding and vector storage using OpenAI Embeddings and ChromaDB
- Conversational RAG with history using LangChain
- Evaluation of generated answers with relevance and groundedness scores
- Streamlit UI and Flask API support

---

## Demo Screenshots

### 1. Streamlit UI – Query and Output
![UI Screenshot](images/ui_query_output.jpg)

### 2. Evaluation Metrics Output
![Evaluation Screenshot](images/evaluation.jpg)

### 3. API Test via Postman
![Postman API Screenshot](images/api_postman.jpg)

---

## Tools & Libraries Used

| Tool/Library       | Purpose                                  |
|--------------------|-------------------------------------------|
| `pdf2image`        | Convert PDF pages to images               |
| `pytesseract`      | OCR for Bangla text extraction            |
| `langchain`        | RAG chain construction and memory         |
| `OpenAIEmbeddings` | Convert chunks into semantic vectors      |
| `Chroma`           | Local vector DB for retrieval             |
| `Streamlit`        | Simple interactive UI                     |
| `Flask`            | Lightweight API backend                   |
| `scikit-learn`     | Evaluation using cosine similarity        |

---

## ⚙️ Setup Guide

1. **Clone the repo**
```bash
https://github.com/sonjoy2074/BanglaBOT
cd BanglaBOT
