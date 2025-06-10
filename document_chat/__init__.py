"""
Document Chat Application
A powerful document chat application with RAG capabilities.
"""

from .src.api import app
from .src import (
    DocumentProcessor,
    VectorStore,
    OllamaClient,
    ChatManager,
    split_text,
    create_faiss_index,
    get_embeddings
)

__version__ = "0.1.0"

__all__ = [
    "app",
    "DocumentProcessor",
    "VectorStore",
    "OllamaClient",
    "ChatManager",
    "split_text",
    "create_faiss_index",
    "get_embeddings",
    "__version__"
] 