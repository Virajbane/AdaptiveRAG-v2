import hashlib
import json
from typing import Optional

from app.services.memory.redis_client import redis_client

CACHE_KEY_PREFIX = "querycache:"


class QueryCache:
    """
    Redis-backed cache for query results with knowledge version tracking.

    Moved off an in-memory dict (Phase 14) because docker-compose.prod.yml
    runs 2 backend replicas - an in-memory cache meant each replica had
    its own separate cache, so a hit on replica A was a miss on replica B.
    Redis is shared across all replicas, so this fixes a correctness gap,
    not just a performance one.

    2026-08-09: Added knowledge_version parameter to cache key scope.
    This prevents stale cached answers when the underlying knowledge base
    (documents, indexes, retrieval sources) changes. Cache key now includes:
    user_id + canonical_question + knowledge_version.
    
    Without versioning, the same question asked after a document upload or
    index rebuild would return a cached answer from before the change, even
    though the underlying retrieval result set is now different.

    Public interface mirrors the old in-memory version (get/set/clear/get_stats),
    but every method is now async since it goes over the network to Redis.
    Callers must add `await`.
    """

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = ttl_seconds

    def _hash_query(
        self, question: str, user_id: str, knowledge_version: str = ""
    ) -> str:
        """
        Create cache key from question + user + knowledge version.

        Args:
            question: The (rewritten/canonical) question text.
            user_id: User identifier.
            knowledge_version: Optional version identifier for the knowledge
                base (e.g., document index version, schema revision). If empty,
                defaults to "".
        
        Returns:
            Redis cache key (deterministic MD5 hash).
        
        Design:
        - user_id scopes cache to a user (different users, different caches)
        - question (canonical form) groups rephrased queries (same semantic
          query should hit same cache entry)
        - knowledge_version differentiates cache entries across document/index
          changes (prevents stale answers after knowledge updates)
        
        If knowledge_version is not available, pass "" (empty string); cache
        will still work correctly, but multiple versions of the knowledge
        base will share cache entries until one expires. This is acceptable
        for apps without explicit versioning.
        """
        key = f"{user_id}:{question}:{knowledge_version}".lower()
        return CACHE_KEY_PREFIX + hashlib.md5(key.encode()).hexdigest()

    async def get(
        self, question: str, user_id: str, knowledge_version: str = ""
    ) -> Optional[dict]:
        """
        Get cached result. Returns None on cache miss or Redis being unavailable.
        
        Args:
            question: The question to look up (should be canonical form
                after RewriterAgent normalization in most cases).
            user_id: User identifier.
            knowledge_version: Optional knowledge base version (should match
                the version used when caching the result).
        
        Returns:
            Cached result dict, or None if not found or Redis unavailable.
        """
        cache_key = self._hash_query(question, user_id, knowledge_version)
        raw = await redis_client.get(cache_key)

        if raw is None:
            return None

        try:
            result = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as e:
            # Corrupt or unexpected cache entry - treat as a miss rather
            # than crashing the request.
            print(f"[CACHE] Failed to decode cached value, treating as miss: {e}")
            return None

        print(f"[CACHE HIT] Question: {question[:50]}...")
        return result

    async def set(
        self,
        question: str,
        user_id: str,
        result: dict,
        knowledge_version: str = "",
    ) -> None:
        """
        Cache a result with TTL. Redis handles expiry natively (SETEX) -
        no manual expires_at bookkeeping needed.
        
        Args:
            question: The (canonical/rewritten) question to cache under.
            user_id: User identifier.
            result: The result dict to cache (must be JSON-serializable).
            knowledge_version: Optional knowledge base version identifier.
                Callers should pass the current version if available; if not,
                pass "" (default). This ensures answers are cache-invalidated
                when the knowledge base changes.
        """
        cache_key = self._hash_query(question, user_id, knowledge_version)
        try:
            payload = json.dumps(result)
        except (TypeError, ValueError) as e:
            # If the result isn't JSON-serializable, skip caching rather
            # than raising - caching is an optimization, not a requirement.
            print(f"[CACHE] Failed to serialize result, skipping cache: {e}")
            return

        success = await redis_client.set(cache_key, payload, expire=self.ttl)
        if success:
            print(f"[CACHE STORED] Key: {cache_key[len(CACHE_KEY_PREFIX):][:8]}...")

    async def clear(self) -> None:
        """
        Clear all cached query results (scans and deletes every key
        under the querycache: prefix).
        """
        if not redis_client.redis:
            return
        try:
            async for key in redis_client.redis.scan_iter(match=f"{CACHE_KEY_PREFIX}*"):
                await redis_client.redis.delete(key)
        except Exception as e:
            print(f"[CACHE] Error clearing cache: {e}")

    async def get_stats(self) -> dict:
        """
        Get cache statistics. total_entries requires a SCAN, so this
        is O(n) over cache keys - fine for occasional stats calls, not
        meant for a hot path.
        
        Returns:
            dict with total_entries and ttl_seconds.
        """
        total = 0
        if redis_client.redis:
            try:
                async for _ in redis_client.redis.scan_iter(
                    match=f"{CACHE_KEY_PREFIX}*"
                ):
                    total += 1
            except Exception as e:
                print(f"[CACHE] Error getting stats: {e}")

        return {
            "total_entries": total,
            "ttl_seconds": self.ttl,
        }


# Global instance
query_cache = QueryCache(ttl_seconds=3600)  # 1 hour TTL