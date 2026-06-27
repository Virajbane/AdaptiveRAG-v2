import logging
from datetime import datetime
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse
import os

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    """Log all HTTP requests and responses"""
    
    async def dispatch(self, request: Request, call_next):
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} - "
            f"Client: {request.client.host if request.client else 'Unknown'}"
        )
        
        try:
            response = await call_next(request)
            
            # Log response
            logger.info(
                f"Response: {request.method} {request.url.path} - "
                f"Status: {response.status_code}"
            )
            
            return response
        
        except Exception as e:
            logger.error(
                f"Error in {request.method} {request.url.path}: {str(e)}",
                exc_info=True
            )
            raise