import httpx
import logging
from typing import Optional, Dict, Any
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

class OllamaClient:
    """
    Client for interacting with Ollama API.
    """
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Base URL for Ollama API
        """
        self.base_url = base_url.rstrip('/')
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging for better debugging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def generate(self,
                      prompt: str,
                      model: str = "llama2",
                      system: Optional[str] = None,
                      context: Optional[str] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Generate text using Ollama.
        
        Args:
            prompt: The prompt to generate from
            model: Model to use (default: llama2)
            system: Optional system prompt
            context: Optional context for the generation
            **kwargs: Additional parameters for Ollama API
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Prepare the full prompt with context if provided
            full_prompt = prompt
            if context:
                full_prompt = f"""Context:
{context}

Question: {prompt}

Answer:"""

            # Prepare the request payload
            payload = {
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                **kwargs
            }

            if system:
                payload["system"] = system

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=60.0
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"Error generating with Ollama: {e}")
            raise

    async def list_models(self) -> Dict[str, Any]:
        """
        List available models.
        
        Returns:
            Dictionary containing available models
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise

    async def check_model(self, model: str) -> bool:
        """
        Check if a model is available.
        
        Args:
            model: Model name to check
            
        Returns:
            True if model is available, False otherwise
        """
        try:
            models = await self.list_models()
            return any(m["name"] == model for m in models.get("models", []))
        except Exception as e:
            logger.error(f"Error checking model: {e}")
            return False 