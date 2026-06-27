import hashlib
import json
from typing import Optional

from app.services.memory.redis_client import redis_client

CACHE_KEY_PREFIX = "querycache:"


class QueryCache:
    """
    Redis-backed cache for query results.

    Moved off an in-memory dict (Phase 14) because docker-compose.prod.yml
    runs 2 backend replicas - an in-memory cache meant each replica had
    its own separate cache, so a hit on replica A was a miss on replica B.
    Redis is shared across all replicas, so this fixes a correctness gap,
    not just a performance one.

    Public interface intentionally mirrors the old in-memory version
    (get/set/clear/get_stats), but every method is now async since it
    goes over the network to Redis. Callers must add `await`.
    """

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = ttl_seconds

    def _hash_query(self, question: str, user_id: str) -> str:
        """Create cache key from question + user"""
        key = f"{user_id}:{question}".lower()
        return CACHE_KEY_PREFIX + hashlib.md5(key.encode()).hexdigest()

    async def get(self, question: str, user_id: str) -> Optional[dict]:
        """Get cached result. Returns None on cache miss or Redis being unavailable."""
        cache_key = self._hash_query(question, user_id)
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

    async def set(self, question: str, user_id: str, result: dict):
        """Cache a result with TTL. Redis handles expiry natively (SETEX) -
        no manual expires_at bookkeeping needed."""
        cache_key = self._hash_query(question, user_id)
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

    async def clear(self):
        """Clear all cached query results (scans and deletes every key
        under the querycache: prefix)."""
        if not redis_client.redis:
            return
        try:
            async for key in redis_client.redis.scan_iter(match=f"{CACHE_KEY_PREFIX}*"):
                await redis_client.redis.delete(key)
        except Exception as e:
            print(f"[CACHE] Error clearing cache: {e}")

    async def get_stats(self) -> dict:
        """Get cache statistics. total_entries requires a SCAN, so this
        is O(n) over cache keys - fine for occasional stats calls, not
        meant for a hot path."""
        total = 0
        if redis_client.redis:
            try:
                async for _ in redis_client.redis.scan_iter(match=f"{CACHE_KEY_PREFIX}*"):
                    total += 1
            except Exception as e:
                print(f"[CACHE] Error getting stats: {e}")

        return {
            "total_entries": total,
            "ttl_seconds": self.ttl,
        }


# Global instance
query_cache = QueryCache(ttl_seconds=3600)  # 1 hour TTL