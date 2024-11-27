import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS #Chroma
from langchain.schema import Document


def load_articles(file_path):
    """
    Load articles from a JSON file.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['articles']

def load_vectorstore(embedding_model_path="all-MiniLM-L6-v2"):
    """
    Load the Chroma vectorstore with precomputed embeddings.
    """
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_path)
    articles = load_articles("./data/articles.json")
    
    # Split articles into chunks
    documents = [
        Document(page_content=article['content'], metadata=article.get('metadata', {}))
        for article in articles
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(
        documents=splits, 
        embedding=embedding_model
        )
    return vectorstore

def retrieve_chunks(vectorstore, query, k=2):
    """
    Retrieve top-k chunks relevant to the query.
    """
    return vectorstore.similarity_search(query, k=k)
