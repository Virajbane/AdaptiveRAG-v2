import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json

class QueryCache:
    """In-memory cache for query results"""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl_seconds
    
    def _hash_query(self, question: str, user_id: str) -> str:
        """Create cache key from question + user"""
        key = f"{user_id}:{question}".lower()
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, question: str, user_id: str) -> Optional[dict]:
        """Get cached result"""
        cache_key = self._hash_query(question, user_id)
        
        if cache_key not in self.cache:
            return None
        
        cached = self.cache[cache_key]
        
        # Check expiry
        if datetime.now() > cached["expires_at"]:
            del self.cache[cache_key]
            return None
        
        print(f"[CACHE HIT] Question: {question[:50]}...")
        return cached["result"]
    
    def set(self, question: str, user_id: str, result: dict):
        """Cache a result"""
        cache_key = self._hash_query(question, user_id)
        self.cache[cache_key] = {
            "result": result,
            "expires_at": datetime.now() + timedelta(seconds=self.ttl),
            "created_at": datetime.now()
        }
        print(f"[CACHE STORED] Key: {cache_key[:8]}...")
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        return {
            "total_entries": len(self.cache),
            "ttl_seconds": self.ttl
        }

# Global instance
query_cache = QueryCache(ttl_seconds=3600)  # 1 hour TTL