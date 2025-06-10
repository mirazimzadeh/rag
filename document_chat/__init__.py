"""
Document Chat - A document-based chat application using Ollama and FAISS.
"""

from .core.document_processor import DocumentProcessor
from .core.vector_store import VectorStore, SearchResult
from .core.llm_client import OllamaClient
from .core.chat_manager import ChatManager, ChatMessage

__version__ = "0.1.0"
__all__ = [
    "DocumentProcessor",
    "VectorStore",
    "SearchResult",
    "OllamaClient",
    "ChatManager",
    "ChatMessage"
] 