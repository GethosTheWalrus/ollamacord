import redis
import logging
import os
import json
from typing import Optional, Any

logger = logging.getLogger(__name__)

class RedisClient:
    _instance: Optional['RedisClient'] = None
    _client: Optional[redis.Redis] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisClient, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            try:
                # Get Redis configuration from environment variables
                host = os.getenv("REDIS_HOST", "localhost")
                port = int(os.getenv("REDIS_PORT", "6379"))
                db = int(os.getenv("REDIS_DB", "0"))
                password = os.getenv("REDIS_PASSWORD")
                
                # Initialize Redis client
                self._client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    decode_responses=True  # Automatically decode responses to strings
                )
                
                # Test connection
                self._client.ping()
                logger.info(f"Redis client initialized successfully at {host}:{port}")
                
                # Get cache duration from environment variable (default: 30 minutes)
                self.cache_duration = int(os.getenv("WIKI_CACHE_DURATION", "1800"))  # 1800 seconds = 30 minutes
                logger.info(f"Wiki page cache duration set to {self.cache_duration} seconds")
                
            except Exception as e:
                logger.error(f"Failed to initialize Redis client: {str(e)}")
                self._client = None
    
    @property
    def client(self):
        return self._client
    
    def is_available(self) -> bool:
        return self._client is not None
    
    def get_cached_page(self, url: str) -> Optional[dict]:
        """Get a cached wiki page if it exists and is not expired."""
        if not self.is_available():
            return None
            
        try:
            cached_data = self._client.get(f"wiki_page:{url}")
            if cached_data:
                logger.info(f"Cache hit for URL: {url}")
                return json.loads(cached_data)
            logger.info(f"Cache miss for URL: {url}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached page: {str(e)}")
            return None
    
    def cache_page(self, url: str, data: dict) -> bool:
        """Cache a wiki page with the configured duration."""
        if not self.is_available():
            return False
            
        try:
            self._client.setex(
                f"wiki_page:{url}",
                self.cache_duration,
                json.dumps(data)
            )
            logger.info(f"Cached page for URL: {url} (expires in {self.cache_duration} seconds)")
            return True
        except Exception as e:
            logger.error(f"Error caching page: {str(e)}")
            return False

# Create a singleton instance
redis_client = RedisClient() 