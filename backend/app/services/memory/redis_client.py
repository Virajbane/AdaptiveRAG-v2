import redis.asyncio as redis
from app.config.settings import settings

class RedisClient:
    """Redis connection manager"""
    
    def __init__(self):
        self.redis = None
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis = await redis.from_url(
                settings.REDIS_URL,
                encoding="utf8",
                decode_responses=True
            )
            print("✅ Connected to Redis")
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            self.redis = None
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()
    
    async def set(self, key: str, value: str, expire: int = 86400):
        """Set key-value with TTL"""
        if not self.redis:
            return False
        try:
            await self.redis.setex(key, expire, value)
            return True
        except Exception as e:
            print(f"Redis set error: {e}")
            return False
    
    async def get(self, key: str) -> str:
        """Get value by key"""
        if not self.redis:
            return None
        try:
            return await self.redis.get(key)
        except Exception as e:
            print(f"Redis get error: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete key"""
        if not self.redis:
            return False
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            print(f"Redis delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        if not self.redis:
            return False
        try:
            return await self.redis.exists(key)
        except Exception as e:
            return False

# Global instance
redis_client = RedisClient()