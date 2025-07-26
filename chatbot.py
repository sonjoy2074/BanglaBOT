from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory  # Correct import here
from dotenv import load_dotenv

load_dotenv()

# Add conversational memory to the chatbot
def get_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

# Function to load the retriever from the Chroma database
def load_retriever(persist_directory="chroma_db"):
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=OpenAIEmbeddings()
    )
    return vectordb.as_retriever()

# Function to build the conversational retrieval chain
def build_chain():
    retriever = load_retriever()
    memory = get_memory()
    llm = ChatOpenAI(temperature=0.3, model="gpt-4o")
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return chain
