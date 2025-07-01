"""Prefix caching optimization for faster inference."""

import hashlib
from typing import Dict, Any, Optional
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class PrefixCache:
    """LRU cache for prefix tokens to speed up repeated inference."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize prefix cache.
        
        Args:
            max_size: Maximum number of entries to cache
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        
    def _hash_prompt(self, prompt: str) -> str:
        """Create a hash key for the prompt."""
        return hashlib.md5(prompt.encode('utf-8')).hexdigest()
        
    def get(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Get cached result for prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Cached result if found, None otherwise
        """
        key = self._hash_prompt(prompt)
        
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            logger.debug(f"Cache hit for prompt hash {key[:8]}")
            return self.cache[key]
        else:
            self.misses += 1
            logger.debug(f"Cache miss for prompt hash {key[:8]}")
            return None
            
    def set(self, prompt: str, result: Dict[str, Any]) -> None:
        """Cache result for prompt.
        
        Args:
            prompt: Input prompt
            result: Generation result to cache
        """
        key = self._hash_prompt(prompt)
        
        # Remove oldest entries if at capacity
        while len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"Evicted cache entry {oldest_key[:8]}")
            
        self.cache[key] = result
        logger.debug(f"Cached result for prompt hash {key[:8]}")
        
    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("Cleared prefix cache")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
        }
        
    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self.cache)
        
    def __contains__(self, prompt: str) -> bool:
        """Check if prompt is cached."""
        key = self._hash_prompt(prompt)
        return key in self.cache
