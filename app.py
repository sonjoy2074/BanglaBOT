import streamlit as st
from chatbot import build_chain

st.set_page_config(page_title="Bangla-English RAG Chatbot")
st.title("📖 Bangla-English RAG Chatbot")

if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = build_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask your question (Bangla or English):")

if query:
    response = st.session_state.chat_chain.invoke({
        "question": query,
        "chat_history": st.session_state.chat_history
    })

    st.session_state.chat_history.append((query, response["answer"]))

    st.write("🧠 **Answer:**", response["answer"])

    with st.expander("📄 Retrieved Context"):
        for i, doc in enumerate(response["source_documents"], 1):
            st.markdown(f"**Chunk {i}:** {doc.page_content.strip()[:500]}...")

if st.session_state.chat_history:
    st.markdown("### 💬 Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")
