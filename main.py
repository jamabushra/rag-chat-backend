from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import uuid
from datetime import datetime
from PyPDF2 import PdfReader
import io
import requests
from langchain_text_splitters import CharacterTextSplitter
from langchain_astradb import AstraDBVectorStore, CollectionVectorServiceOptions
from langchain_core.documents import Document
import ollama

load_dotenv()

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Astra DB configuration from environment variables
ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
COLLECTION_NAME = "rag_documents"

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATOR = "\n"

# Ollama configuration
OLLAMA_MODEL = "llama3.2"  # Adjust to your hosted model
OLLAMA_API_URL = "http://localhost:11434/api/generate"


def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, separator: str = SEPARATOR) -> list[Document]:
    """Split text into chunks and return LangChain Documents."""
    try:
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
        )
        chunks = splitter.split_text(text)
        
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "size": len(chunk.encode("utf-8")),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            ) for i, chunk in enumerate(chunks)
        ]
        return documents
    except Exception as e:
        print(f"Text splitting error: {str(e)}")
        raise

def initialize_vector_store():
    """Initialize AstraDBVectorStore with Astra Vectorize options."""
    try:
        # Assuming Astra Vectorize is enabled on your collection
        # Adjust provider and model_name based on your collection's configuration
        vector_service = CollectionVectorServiceOptions(
            provider="nvidia",  # Example provider; replace with your actual provider
            model_name="NV-Embed-QA",  # Example model; replace with your actual model
            authentication=None,  # Adjust if authentication is required
            parameters={"dimension": 1024}  # Match your collection
        )
        
        vector_store = AstraDBVectorStore(
            token=ASTRA_DB_TOKEN,
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            collection_name=COLLECTION_NAME,
            collection_vector_service_options=vector_service,
        )
        return vector_store
    except Exception as e:
        print(f"Error initializing AstraDBVectorStore: {str(e)}")
        raise

def generate_response_with_ollama(context: str, query: str) -> str:
    try:
        prompt = f"""
        Use the following context to answer the question. If the context is insufficient, say "I don't have enough information to answer that."
        
        Context: {context}
        
        Question: {query}
        
        Answer:
        """
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={"temperature": 0.7}
        )
        return response["response"].strip()
    except Exception as e:
        print(f"Ollama generation error: {str(e)}")
        raise

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Read PDF content
        content_bytes = await file.read()
        pdf_file = io.BytesIO(content_bytes)
        pdf_reader = PdfReader(pdf_file)
        
        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        
        if not text:
            raise ValueError("No text could be extracted from the PDF")
        
        doc_id = str(uuid.uuid4())
        
        # Split text into chunks
        documents = split_text(text)
        
        # Initialize vector store
        vector_store = initialize_vector_store()
        
        # Add metadata to link chunks to parent document
        for doc in documents:
            doc.metadata["parent_id"] = doc_id
            doc.metadata["filename"] = file.filename
            doc.metadata["content_type"] = file.content_type
            doc.metadata["upload_date"] = datetime.now().isoformat()
        
        # Add documents to Astra DB
        if documents:
            print(f"Adding {len(documents)} chunks to Astra DB for document {doc_id}")
            vector_store.add_documents(documents)
            for i in range(len(documents)):
                print(f"Inserted chunk {i+1}/{len(documents)} for document {doc_id}")
        else:
            print("No chunks to add to Astra DB")
        
        # Store a metadata-only document
        metadata_doc = {
            "document_id": doc_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "content": None,
            "metadata": {
                "size": file.size,
                "upload_date": datetime.now().isoformat(),
                "total_chunks": len(documents)
            }
        }
        from astrapy.db import AstraDB
        astra_db = AstraDB(token=ASTRA_DB_TOKEN, api_endpoint=ASTRA_DB_API_ENDPOINT)
        collection = astra_db.collection(COLLECTION_NAME)
        collection.insert_one(metadata_doc)
        
        return {"document_id": doc_id, "filename": file.filename}
    except Exception as e:
        print(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(query: dict):
    try:
        query_text = query["text"]
        vector_store = initialize_vector_store()
        
        # Retrieve relevant documents from Astra DB
        results = vector_store.similarity_search(query_text, k=5)  # Top 5 similar documents
        context = "\n".join([doc.page_content for doc in results])
        
        # Generate response with Ollama
        response = generate_response_with_ollama(context, query_text)
        
        return {"response": response}
    except Exception as e:
        print(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)