'''NLP endpoint using haystack and FastAPI'''

import os
from typing import Optional

from fastapi import FastAPI, Response
from dotenv import load_dotenv
from pydantic import BaseModel
from haystack.schema import Document as HaystackDocument
from haystack.document_stores import OpenSearchDocumentStore
from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline

load_dotenv()
OPENSEARCH_SERVER = os.getenv("OPENSEARCH_SERVER")
OPENSEARCH_SERVER_PORT = os.getenv("OPENSEARCH_SERVER_PORT")
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")

app = FastAPI()

document_store = OpenSearchDocumentStore(
    host=OPENSEARCH_SERVER,
    port=OPENSEARCH_SERVER_PORT,
    username=OPENSEARCH_USER,
    password=OPENSEARCH_PASSWORD,
    verify_certs=True
)

class Document(BaseModel):
    '''Document model for REST API request'''
    name: Optional[str] = None
    content: str

class Query(BaseModel):
    '''Query model for REST API request'''
    question: str

@app.get("/health")
def health():
    '''Health check endpoint'''
    return {"status": "ok"}

@app.get("/documents/{document_id}", status_code=200)
def get_document(document_id: str, response: Response):
    '''Get document by ID'''
    document = document_store.get_document_by_id(document_id)
    if document is None:
        response.status_code = 404
        return {"error": "Document not found"}
    return document

@app.post("/documents/", status_code=201)
def save_document(document: Document):
    '''Save document'''
    new_doc= HaystackDocument(content=document.content, meta={"name": document.name})
    document_store.write_documents([new_doc])
    return {"id": new_doc.id}

@app.get("/documents/", status_code=200)
def get_all_document(response: Response):
    '''Get all documents'''
    documents = document_store.get_all_documents()
    if documents is None:
        response.status_code = 404
        return {"error": "Documents not found"}
    return documents

@app.get("/documents/search/{query}", status_code=200)
def search_document(query: str, response: Response):
    '''Search documents by a keyword query'''
    documents = document_store.query(query)
    return documents

@app.post("/documents/ask", status_code=200)
def ask_document(query: Query, response: Response): 
    '''Extract the answer from the documents for a question'''  
    model = "deepset/roberta-base-squad2"
    retriever = BM25Retriever(document_store)
    reader = FARMReader(model, use_gpu=True)
    pipeline = ExtractiveQAPipeline(reader, retriever)
    result = pipeline.run(query=query.question, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 1}})    
    answers = [x.to_dict() for x in result["answers"]]
    return answers