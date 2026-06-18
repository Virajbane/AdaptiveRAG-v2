from datetime import datetime, timedelta
from typing import Optional
import jwt
from app.config.settings import settings

class JWTService:
    """JWT token creation and validation"""
    
    @staticmethod
    def create_access_token(user_id: str, expires_in_hours: int = 24) -> str:
        """Create a JWT access token"""
        expire = datetime.utcnow() + timedelta(hours=expires_in_hours)
        payload = {
            "user_id": user_id,
            "exp": expire,
            "iat": datetime.utcnow()
        }
        token = jwt.encode(
            payload,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )
        return token
    
    @staticmethod
    def create_refresh_token(user_id: str) -> str:
        """Create a refresh token (7 days expiry)"""
        expire = datetime.utcnow() + timedelta(days=7)
        payload = {
            "user_id": user_id,
            "type": "refresh",
            "exp": expire,
            "iat": datetime.utcnow()
        }
        token = jwt.encode(
            payload,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )
        return token
    
    @staticmethod
    def verify_token(token: str) -> dict:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.ALGORITHM]
            )
            user_id = payload.get("user_id")
            if user_id is None:
                raise ValueError("Invalid token")
            return {"user_id": user_id, "valid": True}
        except jwt.ExpiredSignatureError:
            raise ValueError("Token expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")