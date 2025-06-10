import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from rich.console import Console

from .vector_store import VectorStore, SearchResult
from .llm_client import OllamaClient

console = Console()
logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Structure for chat messages"""
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class ChatManager:
    """
    Manages chat interactions using vector store and LLM.
    """
    def __init__(self,
                 vector_store: VectorStore,
                 llm_client: OllamaClient,
                 model: str = "llama2",
                 system_prompt: Optional[str] = None,
                 max_context_chunks: int = 5,
                 similarity_threshold: float = 0.7):
        """
        Initialize chat manager.
        
        Args:
            vector_store: Vector store instance
            llm_client: LLM client instance
            model: Model to use for generation
            system_prompt: Optional system prompt
            max_context_chunks: Maximum number of chunks to include in context
            similarity_threshold: Minimum similarity score for context chunks
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.model = model
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_context_chunks = max_context_chunks
        self.similarity_threshold = similarity_threshold
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging for better debugging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _default_system_prompt(self) -> str:
        """Get default system prompt"""
        return """You are a helpful AI assistant that answers questions based on the provided context.
If the context doesn't contain enough information to answer the question, say so.
Always be truthful and don't make up information.
If you're not sure about something, say so."""

    def _format_context(self, results: List[SearchResult]) -> str:
        """
        Format search results into context string.
        
        Args:
            results: List of search results
            
        Returns:
            Formatted context string
        """
        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"Context {i} (Relevance: {result.score:.2f}):")
            context_parts.append(result.text)
            context_parts.append("")

        return "\n".join(context_parts)

    async def get_response(self,
                          message: str,
                          history: Optional[List[ChatMessage]] = None) -> ChatMessage:
        """
        Get response for a message.
        
        Args:
            message: User message
            history: Optional chat history
            
        Returns:
            ChatMessage containing the response
        """
        try:
            # Search for relevant context
            results = self.vector_store.search(
                query=message,
                top_k=self.max_context_chunks,
                similarity_threshold=self.similarity_threshold
            )

            # Format context
            context = self._format_context(results)

            # Generate response
            response = await self.llm_client.generate(
                prompt=message,
                model=self.model,
                system=self.system_prompt,
                context=context
            )

            # Create response message
            return ChatMessage(
                role="assistant",
                content=response["response"],
                metadata={
                    "model": self.model,
                    "context_chunks": len(results),
                    "generation_time": response.get("total_duration", 0)
                }
            )

        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return ChatMessage(
                role="assistant",
                content="I apologize, but I encountered an error while processing your request. Please try again.",
                metadata={"error": str(e)}
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get chat manager statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "model": self.model,
            "max_context_chunks": self.max_context_chunks,
            "similarity_threshold": self.similarity_threshold,
            "vector_store_stats": self.vector_store.get_stats()
        } 