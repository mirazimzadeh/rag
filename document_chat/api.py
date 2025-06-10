import os
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager
import shutil

from .core.document_processor import DocumentProcessor
from .core.vector_store import VectorStore
from .core.llm_client import OllamaClient
from .core.chat_manager import ChatManager, ChatMessage
from .config import (
    UPLOAD_DIR,
    VECTOR_STORE_DIR,
    STATIC_DIR,
    OLLAMA_CONFIG,
    RAG_CONFIG,
    VECTOR_STORE_CONFIG
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Chat API",
    description="A document-based chat application using Ollama and FAISS",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)

# Initialize components
document_processor = DocumentProcessor(UPLOAD_DIR)
vector_store = VectorStore(
    VECTOR_STORE_DIR,
    chunk_size=VECTOR_STORE_CONFIG["chunk_size"],
    chunk_overlap=VECTOR_STORE_CONFIG["chunk_overlap"],
    embedding_model=VECTOR_STORE_CONFIG["embedding_model"],
    similarity_metric=VECTOR_STORE_CONFIG["similarity_metric"]
)
llm_client = OllamaClient(
    base_url=OLLAMA_CONFIG["base_url"],
    model=OLLAMA_CONFIG["default_model"]
)
chat_manager = ChatManager(
    vector_store,
    llm_client,
    max_context_chunks=RAG_CONFIG["default_max_chunks"],
    similarity_threshold=RAG_CONFIG["default_similarity_threshold"]
)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Models
class RAGParams(BaseModel):
    max_chunks: int = RAG_CONFIG["max_chunks"]
    similarity_threshold: float = RAG_CONFIG["similarity_threshold"]
    model: str = OLLAMA_CONFIG["default_model"]

class ChatRequest(BaseModel):
    message: str
    rag_params: Optional[dict] = None

class ChatResponse(BaseModel):
    message: ChatMessage
    stats: dict

class StatsResponse(BaseModel):
    chat_stats: dict
    vector_store_stats: dict

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

# Routes
@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the chat interface"""
    try:
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Chat interface not found")
        return index_path.read_text()
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process documents."""
    try:
        # Save uploaded files
        saved_files = []
        for file in files:
            file_path = Path(UPLOAD_DIR) / file.filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)
            
        # Process documents
        documents = document_processor.scan_documents(UPLOAD_DIR)
        
        # Add to vector store
        vector_store.add_documents(documents)
        
        return JSONResponse({
            "message": f"Successfully processed {len(documents)} documents",
            "documents": [doc["metadata"] for doc in documents]
        })
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    """Process a chat message with optional RAG."""
    try:
        # Get RAG parameters
        rag_params = RAGParams(**(request.rag_params or {}))
        
        # Get relevant context if RAG is enabled
        context = ""
        if request.rag_params:
            results = vector_store.search(
                request.message,
                k=rag_params.max_chunks,
                similarity_threshold=rag_params.similarity_threshold
            )
            if results:
                context = "\n\n".join(r["text"] for r in results)
                
        # Generate response
        response = llm_client.generate(
            request.message,
            context=context,
            model=rag_params.model
        )
        
        return JSONResponse({
            "message": {
                "content": response,
                "metadata": {
                    "used_rag": bool(request.rag_params),
                    "context_sources": [r["metadata"] for r in results] if request.rag_params else []
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get statistics about the vector store."""
    try:
        return JSONResponse({
            "vector_store_stats": vector_store.get_stats()
        })
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available Ollama models"""
    try:
        return await llm_client.list_models()
    except Exception as e:
        logger.error(f"Models error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 