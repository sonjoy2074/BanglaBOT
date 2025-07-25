from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from chatbot import build_chain
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize the embedding model
embedding_model = OpenAIEmbeddings()

# Get embedding vector
def get_embedding(text):
    return embedding_model.embed_query(text)

# Score: How well the answer is grounded in retrieved context
def groundedness_score(answer, retrieved_context):
    ans_vec = get_embedding(answer)
    ctx_vec = get_embedding(retrieved_context)
    score = cosine_similarity([ans_vec], [ctx_vec])[0][0]
    return score

# Score: How relevant are the retrieved chunks to the query
def relevance_score(query, retrieved_chunks):
    query_vec = get_embedding(query)
    scores = []
    for chunk in retrieved_chunks:
        chunk_vec = get_embedding(chunk)
        sim = cosine_similarity([query_vec], [chunk_vec])[0][0]
        scores.append(sim)
    return np.mean(scores) if scores else 0.0

def run_evaluation():
    chain = build_chain()
    
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        response = chain.invoke({"question": query})
        answer = response.get("answer", "")
        docs = response.get("source_documents", [])

        # Extract context text from documents
        context_chunks = [doc.page_content for doc in docs]
        combined_context = " ".join(context_chunks)

        # Compute metrics
        g_score = groundedness_score(answer, combined_context)
        r_score = relevance_score(query, context_chunks)

        # Display result
        print("\nAnswer:")
        print(answer)
        print("\nRetrieved Context:")
        for i, doc in enumerate(docs, 1):
            print(f"\n--- Chunk {i} ---\n{doc.page_content.strip()[:500]}")  # limit long text

        print(f"\nEvaluation Scores:")
        print(f"Groundedness: {g_score:.3f}")
        print(f"Relevance: {r_score:.3f}")

if __name__ == "__main__":
    run_evaluation()
