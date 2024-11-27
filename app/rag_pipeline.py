from app.vectorstore import load_vectorstore, retrieve_chunks
from app.generator import LLMClient

llm_client = LLMClient(api_key="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")
vectorstore = load_vectorstore()

def get_rag_response(query):
    """
    RAG pipeline: Retrieve context and generate a response.
    """
    retrieved_docs = retrieve_chunks(vectorstore, query, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    response = llm_client.get_response_from_model(context, query)
    return {
        "query": query,
        "response": response,
        "context": [doc.page_content for doc in retrieved_docs]
    }
