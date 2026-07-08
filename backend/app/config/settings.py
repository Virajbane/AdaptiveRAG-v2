from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # ==================== APP ====================
    APP_NAME: str = "RAG 2.0 System"
    DEBUG: bool = True

    # ==================== DATABASE ====================
    MONGODB_URL: str  # Required from .env
    QDRANT_URL: str = "http://localhost:6333"
    REDIS_URL: str = "redis://localhost:6379/0"
    QDRANT_API_KEY: Optional[str] = None


    # ==================== LANGSMITH ====================
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "rag-2.0-system"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"

    # ==================== LLM ====================
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"
    OLLAMA_FAST_MODEL: str = "qwen2.5:0.5b"   
    EMBEDDING_MODEL: str = "nomic-embed-text"
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # ==================== API KEYS ====================
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    TAVILY_API_KEY: str = ""

    # ==================== SECURITY ====================
    SECRET_KEY: str  # Required from .env
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_HOURS: int = 24
    
    # NEW PHASE 10 SECURITY FIELDS
    ENCRYPTION_KEY: str = "default-change-in-production"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_LOWERCASE: bool = True
    PASSWORD_REQUIRE_DIGITS: bool = True
    PASSWORD_REQUIRE_SPECIAL: bool = True

    # ==================== CORS ====================
    FRONTEND_URL: str = "http://localhost:3000"
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000"

    class Config:
        env_file = ".env"
        extra = "allow"  # Allow extra fields from .env

settings = Settings()