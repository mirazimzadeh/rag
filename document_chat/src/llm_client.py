import httpx
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma:2b-it-qat",
        timeout: int = 30
    ):
        """Initialize the Ollama client.
        
        Args:
            base_url: Base URL for Ollama API
            model: Default model to use
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        
    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text using the Ollama model.
        
        Args:
            prompt: Input prompt
            context: Optional context for RAG
            model: Model to use (defaults to instance model)
            **kwargs: Additional model parameters
            
        Returns:
            Generated text
        """
        try:
            # Prepare the prompt with context if provided
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            
            # Prepare the request
            data = {
                "model": model or self.model,
                "prompt": full_prompt,
                "stream": False,
                **kwargs
            }
            
            # Make the request
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json=data
            )
            response.raise_for_status()
            
            # Parse and return the response
            result = response.json()
            return result["response"]
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
            
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models.
        
        Returns:
            Dictionary containing model information
        """
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            raise 