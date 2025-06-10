import logging
from typing import List, Dict, Any, Optional
from .llm_client import OllamaClient
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

class ChatManager:
    """Manages chat interactions with RAG capabilities."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: OllamaClient,
        max_context_chunks: int = 3,
        similarity_threshold: float = 0.7
    ):
        """Initialize the chat manager.
        
        Args:
            vector_store: Vector store instance for document retrieval
            llm_client: LLM client for text generation
            max_context_chunks: Maximum number of context chunks to use
            similarity_threshold: Minimum similarity score for context chunks
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.max_context_chunks = max_context_chunks
        self.similarity_threshold = similarity_threshold
        
    def get_response(
        self,
        query: str,
        use_rag: bool = True,
        max_chunks: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get a response for a user query.
        
        Args:
            query: User's query
            use_rag: Whether to use RAG for context
            max_chunks: Override for max context chunks
            similarity_threshold: Override for similarity threshold
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Get relevant context if RAG is enabled
            context = None
            metadata = {}
            
            if use_rag:
                # Use provided parameters or defaults
                chunks = max_chunks or self.max_context_chunks
                threshold = similarity_threshold or self.similarity_threshold
                
                # Search for relevant documents
                results = self.vector_store.search(
                    query,
                    k=chunks,
                    similarity_threshold=threshold
                )
                
                if results:
                    # Combine context from results
                    context = "\n".join(r["text"] for r in results)
                    metadata = {
                        "num_chunks": len(results),
                        "similarity_scores": [r["score"] for r in results],
                        "sources": [r["metadata"].get("source", "Unknown") for r in results]
                    }
            
            # Generate response
            response = self.llm_client.generate(
                prompt=query,
                context=context
            )
            
            return {
                "response": response,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            raise 