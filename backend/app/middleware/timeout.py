import asyncio
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

class TimeoutMiddleware(BaseHTTPMiddleware):
    """Add request timeouts to prevent hanging"""
    
    TIMEOUT_CONFIG = {
        "/api/v1/agents/chat": 120,        # 2 minutes max
        "/api/v1/documents/upload": 300,   # 5 minutes max
        "/api/v1/auth/login": 30,          # 30 seconds max
        "/api/v1/auth/register": 30,       # 30 seconds max
    }
    
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        timeout = None
        
        # Find matching timeout
        for endpoint, seconds in self.TIMEOUT_CONFIG.items():
            if path.startswith(endpoint):
                timeout = seconds
                break
        
        if not timeout:
            return await call_next(request)
        
        try:
            # Run request with timeout
            response = await asyncio.wait_for(
                call_next(request),
                timeout=timeout
            )
            return response
        
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={
                    "detail": f"Request timeout after {timeout} seconds",
                    "error": "GATEWAY_TIMEOUT"
                }
            )