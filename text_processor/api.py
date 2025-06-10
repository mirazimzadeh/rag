from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx
import json
from pathlib import Path
import logging
from rich.console import Console
import shutil
import os

from .vector_db import VectorDB
from .scanner import Scanner

console = Console()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Chat API",
    description="API for chatting with documents using RAG and Ollama",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="text_processor/static"), name="static")

# Initialize components
vector_db = VectorDB()
scanner = None  # Will be initialized when a folder is set

# Models
class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    similarity_threshold: float = 0.0
    model: str = "llama2"  # Default Ollama model

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class ProcessFolderRequest(BaseModel):
    folder_path: str
    extensions: Optional[List[str]] = None

class ProcessFolderResponse(BaseModel):
    status: str
    message: str
    stats: Optional[Dict[str, Any]] = None

# Ollama client
async def get_ollama_response(prompt: str, model: str = "llama2") -> str:
    """Get response from Ollama API"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"]
    except Exception as e:
        logger.error(f"Error getting response from Ollama: {e}")
        raise HTTPException(status_code=500, detail="Failed to get response from Ollama")

def process_documents(folder_path: str, extensions: Optional[List[str]] = None):
    """Process documents in background"""
    global scanner
    try:
        scanner = Scanner(folder_path, extensions)
        for file_path in scanner.search_files_with_progress():
            text = scanner.extract_text(file_path)
            if text.strip():
                vector_db.add(text)
        return True
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        return False

@app.post("/process-folder", response_model=ProcessFolderResponse)
async def process_folder(request: ProcessFolderRequest, background_tasks: BackgroundTasks):
    """Process documents in a folder"""
    try:
        folder_path = Path(request.folder_path)
        if not folder_path.exists():
            raise HTTPException(status_code=404, detail="Folder not found")

        background_tasks.add_task(
            process_documents,
            str(folder_path),
            request.extensions
        )

        return ProcessFolderResponse(
            status="processing",
            message="Document processing started in background"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=Dict[str, Any])
async def get_stats():
    """Get vector database statistics"""
    return vector_db.get_stats()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the documents"""
    try:
        # Search for relevant documents
        results = vector_db.search(
            request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )

        if not results:
            return ChatResponse(
                response="I couldn't find any relevant information in the documents.",
                sources=[],
                metadata={"status": "no_results"}
            )

        # Prepare context from search results
        context = "\n\n".join([f"Source {i+1}:\n{r.text}" for i, r in enumerate(results)])
        
        # Create prompt with context
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.
        Use only the information from the context to answer the question.
        If the context doesn't contain relevant information, say so.

        Context:
        {context}

        Question: {request.query}

        Answer:"""

        # Get response from Ollama
        response = await get_ollama_response(prompt, request.model)

        # Prepare sources
        sources = [
            {
                "text": r.text,
                "score": r.score,
                "metadata": r.metadata
            }
            for r in results
        ]

        return ChatResponse(
            response=response,
            sources=sources,
            metadata={
                "model": request.model,
                "num_sources": len(sources),
                "avg_score": sum(s["score"] for s in sources) / len(sources) if sources else 0
            }
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    """Serve the main page"""
    return FileResponse("text_processor/static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 