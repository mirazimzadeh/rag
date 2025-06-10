"""
Document Chat - A powerful document chat application with RAG capabilities.
"""

from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .llm_client import OllamaClient
from .utils import split_text, create_faiss_index, get_embeddings

__version__ = "0.1.0"

__all__ = [
    'DocumentProcessor',
    'VectorStore',
    'OllamaClient',
    'split_text',
    'create_faiss_index',
    'get_embeddings',
    '__version__'
] 