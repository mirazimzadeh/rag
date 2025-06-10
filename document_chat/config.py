"""
Configuration settings for the Document Chat application.
"""

from pathlib import Path
from typing import Dict, Any

# Base directories
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
STATIC_DIR = BASE_DIR / "static"

# Create necessary directories
for directory in [UPLOAD_DIR, VECTOR_STORE_DIR, STATIC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Ollama settings
OLLAMA_CONFIG: Dict[str, Any] = {
    "base_url": "http://localhost:11434",
    "default_model": "gemma:2b-it-qat",
    "available_models": [
        "gemma:2b-it-qat",
        "llama2",
        "mistral",
        "codellama"
    ],
    "model_params": {
        "gemma:2b-it-qat": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048
        },
        "llama2": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048
        },
        "mistral": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048
        },
        "codellama": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048
        }
    }
}

# Vector store settings
VECTOR_STORE_CONFIG: Dict[str, Any] = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "similarity_metric": "cosine"
}

# RAG settings
RAG_CONFIG: Dict[str, Any] = {
    "default_max_chunks": 5,
    "default_similarity_threshold": 0.7,
    "max_chunks_range": (1, 20),
    "similarity_threshold_range": (0.0, 1.0)
} 