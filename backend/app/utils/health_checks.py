import app.db.mongodb.client as mongo_client
from app.services.memory.redis_client import redis_client
from datetime import datetime
from app.config.settings import settings
import httpx

async def check_mongodb() -> dict:
    """Check MongoDB connection"""
    try:
        if mongo_client.db is not None:
            await mongo_client.db.command("ping")
            return {"status": "healthy", "service": "mongodb"}

        return {
            "status": "unhealthy",
            "service": "mongodb",
            "error": "Not initialized"
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "mongodb",
            "error": str(e)
        }

async def check_redis() -> dict:
    """Check Redis connection"""
    try:
        if redis_client.redis:
            await redis_client.redis.ping()
            return {"status": "healthy", "service": "redis"}
        return {"status": "unhealthy", "service": "redis", "error": "Not initialized"}
    except Exception as e:
        return {"status": "unhealthy", "service": "redis", "error": str(e)}

async def check_qdrant() -> dict:
    """Check Qdrant connection"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.QDRANT_URL}/collections",
                timeout=5.0
            )
            if response.status_code == 200:
                return {"status": "healthy", "service": "qdrant"}
            return {"status": "unhealthy", "service": "qdrant", "error": "Bad status"}
    except Exception as e:
        return {"status": "unhealthy", "service": "qdrant", "error": str(e)}

async def check_ollama() -> dict:
    """Check Ollama connection"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.OLLAMA_BASE_URL}/api/tags",
                timeout=5.0
            )
            if response.status_code == 200:
                return {"status": "healthy", "service": "ollama"}
            return {"status": "unhealthy", "service": "ollama", "error": "Bad status"}
    except Exception as e:
        return {"status": "unhealthy", "service": "ollama", "error": str(e)}

async def get_system_health() -> dict:
    """Get overall system health"""
    checks = [
        await check_mongodb(),
        await check_redis(),
        await check_qdrant(),
        await check_ollama()
    ]
    
    all_healthy = all(check["status"] == "healthy" for check in checks)
    
    return {
        "overall": "healthy" if all_healthy else "degraded",
        "services": checks,
        "timestamp": datetime.now().isoformat()
    }