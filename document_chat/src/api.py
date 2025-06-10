import os
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .llm_client import OllamaClient
from .chat_manager import ChatManager
from .config import (
    UPLOAD_DIR,
    VECTOR_STORE_DIR,
    STATIC_DIR,
    OLLAMA_CONFIG,
    RAG_CONFIG,
    VECTOR_STORE_CONFIG
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Document Chat API")

# Create necessary directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Initialize components
document_processor = DocumentProcessor(upload_dir=UPLOAD_DIR)
vector_store = VectorStore(
    store_dir=VECTOR_STORE_DIR,
    chunk_size=VECTOR_STORE_CONFIG["chunk_size"],
    chunk_overlap=VECTOR_STORE_CONFIG["chunk_overlap"],
    embedding_model=VECTOR_STORE_CONFIG["embedding_model"],
    similarity_metric=VECTOR_STORE_CONFIG["similarity_metric"]
)
llm_client = OllamaClient(
    base_url=OLLAMA_CONFIG["base_url"],
    model=OLLAMA_CONFIG["model"]
)
chat_manager = ChatManager(
    vector_store=vector_store,
    llm_client=llm_client,
    max_context_chunks=RAG_CONFIG["max_chunks"],
    similarity_threshold=RAG_CONFIG["similarity_threshold"]
)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Models
class ChatRequest(BaseModel):
    query: str
    use_rag: bool = True
    max_chunks: int = RAG_CONFIG["max_chunks"]
    similarity_threshold: float = RAG_CONFIG["similarity_threshold"]

class ChatResponse(BaseModel):
    response: str
    metadata: Dict[str, Any]

# Routes
@app.post("/upload", response_model=List[Dict[str, Any]])
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process documents."""
    try:
        processed_docs = []
        for file in files:
            # Save file
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Process document
            docs = document_processor.scan_documents(file_path)
            processed_docs.extend(docs)
            
            # Add to vector store
            vector_store.add_documents(docs)
            
        return processed_docs
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Get chat response."""
    try:
        response = chat_manager.get_response(
            query=request.query,
            use_rag=request.use_rag,
            max_chunks=request.max_chunks,
            similarity_threshold=request.similarity_threshold
        )
        return response
    except Exception as e:
        logger.error(f"Error getting chat response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get vector store statistics."""
    try:
        return vector_store.get_stats()
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    """Get available Ollama models."""
    try:
        return llm_client.get_available_models()
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 