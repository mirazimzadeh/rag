import os
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager

from .core.document_processor import DocumentProcessor
from .core.vector_store import VectorStore
from .core.llm_client import OllamaClient
from .core.chat_manager import ChatManager, ChatMessage

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
UPLOAD_DIR = Path("uploads")
VECTOR_STORE_DIR = Path("vector_store")
STATIC_DIR = Path("static")

for directory in [UPLOAD_DIR, VECTOR_STORE_DIR, STATIC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Initialize components
document_processor = DocumentProcessor(upload_dir=str(UPLOAD_DIR))
vector_store = VectorStore(store_dir=str(VECTOR_STORE_DIR))
llm_client = OllamaClient()
chat_manager = ChatManager(vector_store, llm_client)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Models
class RAGParams(BaseModel):
    max_chunks: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    model: str = Field(default="llama2")

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: Optional[List[ChatMessage]] = None
    rag_params: Optional[RAGParams] = None

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
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process documents"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        # Save uploaded files
        saved_files = []
        for file in files:
            if not file.filename:
                continue
            file_path = UPLOAD_DIR / file.filename
            try:
                content = await file.read()
                with open(file_path, "wb") as f:
                    f.write(content)
                saved_files.append(str(file_path))
            except Exception as e:
                logger.error(f"Error saving file {file.filename}: {e}")
                continue

        if not saved_files:
            raise HTTPException(status_code=400, detail="No valid files were uploaded")

        # Process documents
        texts, metadata = document_processor.scan_documents(saved_files)
        
        if not texts:
            raise HTTPException(status_code=400, detail="No text could be extracted from the documents")
        
        # Add to vector store
        vector_store.add_documents(texts, source_metadata=metadata)
        
        return {
            "message": f"Successfully processed {len(texts)} documents",
            "stats": vector_store.get_stats()
        }
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat message and return response"""
    try:
        # Update chat manager parameters if RAG is enabled
        if request.rag_params:
            chat_manager.max_context_chunks = request.rag_params.max_chunks
            chat_manager.similarity_threshold = request.rag_params.similarity_threshold
            chat_manager.model = request.rag_params.model

        # Get response from chat manager
        response = await chat_manager.get_response(
            message=request.message,
            history=request.history,
            use_rag=request.rag_params is not None
        )

        return ChatResponse(
            message=response,
            stats=chat_manager.get_stats()
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    try:
        return StatsResponse(
            chat_stats=chat_manager.get_stats(),
            vector_store_stats=vector_store.get_stats()
        )
    except Exception as e:
        logger.error(f"Stats error: {e}")
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