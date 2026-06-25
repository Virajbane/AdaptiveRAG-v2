from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from datetime import datetime, timedelta
from collections import defaultdict
import json

class RateLimitStore:
    """In-memory rate limit store"""
    
    def __init__(self):
        self.requests = defaultdict(list)
    
    async def check_rate_limit(
        self,
        key: str,
        max_requests: int = 100,
        window_seconds: int = 60
    ) -> bool:
        """Check if request should be allowed"""
        now = datetime.now()
        cutoff = now - timedelta(seconds=window_seconds)
        
        # Remove old requests
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if req_time > cutoff
        ]
        
        # Check limit
        if len(self.requests[key]) >= max_requests:
            return False
        
        # Add current request
        self.requests[key].append(now)
        return True

# Global instance
rate_limit_store = RateLimitStore()

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limit requests by IP address"""
    
    LIMITS = {
        "/api/v1/auth/login": {"max": 5, "window": 300},      # 5 per 5 min
        "/api/v1/auth/register": {"max": 3, "window": 3600},  # 3 per hour
        "/api/v1/agents/chat": {"max": 100, "window": 3600},  # 100 per hour
        "/api/v1/tools/web-search": {"max": 20, "window": 3600},  # 20 per hour
    }
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limits
        for endpoint, limit_config in self.LIMITS.items():
            if request.url.path.startswith(endpoint):
                allowed = await rate_limit_store.check_rate_limit(
                    f"{client_ip}:{endpoint}",
                    max_requests=limit_config["max"],
                    window_seconds=limit_config["window"]
                )
                
                if not allowed:
                    # Return 429 response directly (don't raise exception)
                    return JSONResponse(
                        status_code=429,
                        content={
                            "detail": "Rate limit exceeded. Try again later.",
                            "error": "RATE_LIMIT_EXCEEDED"
                        }
                    )
        
        # Continue to next middleware
        response = await call_next(request)
        return response