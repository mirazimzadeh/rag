import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Ollama configuration
OLLAMA_CONFIG = {
    "base_url": "http://localhost:11434",
    "model": "gemma:2b-it-qat",
    "timeout": 30
}

# RAG configuration
RAG_CONFIG = {
    "max_chunks": 3,
    "similarity_threshold": 0.7
}

# Vector store configuration
VECTOR_STORE_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "all-MiniLM-L6-v2",
    "similarity_metric": "cosine"
} 