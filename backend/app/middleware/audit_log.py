from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Log all important actions for audit trail"""
    
    # Actions to audit
    AUDIT_PATHS = [
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/v1/documents",
        "/api/v1/memory",
    ]
    
    async def dispatch(self, request: Request, call_next):
        should_audit = any(
            request.url.path.startswith(path)
            for path in self.AUDIT_PATHS
        )
        
        if should_audit:
            # Log request
            audit_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown")
            }
            
            logger.info(f"AUDIT: {audit_entry}")
        
        response = await call_next(request)
        
        if should_audit:
            logger.info(f"AUDIT RESPONSE: {request.url.path} - {response.status_code}")
        
        return response