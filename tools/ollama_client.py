import logging
import ollama
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class OllamaClient:
    _instance: Optional['OllamaClient'] = None
    _client: Optional[Any] = None
    
    # Default model names that can be overridden by environment variables
    DEFAULT_CHAT_MODEL = "llama2"  # Default chat model
    DEFAULT_SUMMARY_MODEL = "llama2"  # Default model for summaries
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OllamaClient, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            try:
                host = os.getenv("OLLAMA_URL", "http://localhost:11434")
                self._client = ollama.Client(host=host)
                logger.info(f"Ollama client initialized successfully at {host}")
                
                # Get model names from environment variables or use defaults
                self.chat_model = os.getenv("OLLAMA_CHAT_MODEL", self.DEFAULT_CHAT_MODEL)
                self.summary_model = os.getenv("OLLAMA_SUMMARY_MODEL", self.DEFAULT_SUMMARY_MODEL)
                logger.info(f"Using chat model: {self.chat_model}")
                logger.info(f"Using summary model: {self.summary_model}")
            except Exception as e:
                logger.error(f"Failed to initialize Ollama client: {str(e)}")
                self._client = None
    
    @property
    def client(self):
        return self._client
    
    def is_available(self) -> bool:
        return self._client is not None
    
    def chat(self, model: str = None, messages: list = None, stream: bool = False, **kwargs) -> Dict:
        """
        Wrapper around ollama.Client.chat with error handling
        """
        if not self.is_available():
            raise ValueError("Ollama client is not available")
        
        # Use provided model or default to chat_model
        model_to_use = model or self.chat_model
        
        try:
            return self._client.chat(
                model=model_to_use,
                messages=messages,
                stream=stream,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in Ollama chat: {str(e)}")
            raise

# Create a singleton instance
ollama_client = OllamaClient() 