from fastapi import FastAPI
from app.models import QueryRequest, QueryResponse
from app.rag_pipeline import get_rag_response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Welcome to the RAG backend API!"}

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Handle user queries and return the RAG-generated response.
    """
    response = get_rag_response(request.query)
    return response
