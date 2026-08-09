# PATH: backend/app/config/settings.py
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
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    ENABLE_RERANKER: bool = True

    # ==================== DOCUMENT PROCESSING ====================
    # Pages per mini-PDF batch when parsing via Docling. Bounds Docling's
    # peak memory per convert() call (~this many pages' worth of
    # layout/table analysis at once) instead of scaling with total
    # document size -- fixes std::bad_alloc on large/complex PDFs.
    # Tune down (e.g. 3) on smaller/memory-constrained environments.
    DOCLING_BATCH_SIZE: int = 5

    # Vision model used by Docling's picture-description pass to transcribe
    # chart/figure content (e.g. bar chart values) into searchable text.
    # 2026-07-14 fix (Bug 3, figure-value extraction): figures were
    # previously dropped entirely at chunking -- DoclingChunker had no
    # branch for PICTURE items, and no description existed to chunk in the
    # first place since do_picture_description was never enabled. Kept as
    # a setting (not hardcoded) so it can be swapped without a code change
    # if a given model proves too weak on dense charts (e.g. try
    # "qwen2.5vl:3b" if "moondream" undershoots on the UTMOS bar chart).
    #
    # 2026-08-03 fix (Q18 root cause, confirmed via debug_vision_model_direct.py):
    # moondream (1.6B) is a lightweight captioning model -- confirmed by direct
    # isolated testing (correct image, correct payload/content-block shape,
    # correct resolution) to hallucinate unrelated narratives on dense bar
    # charts (e.g. described the UTMOS chart as "an excel graph of user test
    # scores") instead of transcribing labeled values, and does not reliably
    # follow structured "label: value" output instructions. Swapped to
    # qwen2.5vl:3b (not 7b -- dev machine has a 2GB VRAM GPU, 7b won't fit).
    # Requires `ollama pull qwen2.5vl:3b` before this takes effect. Expect
    # this to run partly/mostly on CPU via Ollama's automatic VRAM/RAM
    # offload -- slow (seconds, possibly tens of seconds per figure) but
    # acceptable since do_picture_description only runs once per document
    # at ingestion time, not per chat query. If precision on individual
    # digits is still off at 3b, consider routing this one step to a
    # hosted vision API (e.g. gpt-4o-mini) instead of a larger local model,
    # since VRAM is the hard constraint here, not just this specific model.
    ENABLE_PICTURE_DESCRIPTION: bool = True
    PICTURE_DESCRIPTION_MODEL: str = "qwen2.5vl:3b"

    LOGO_STRIP_ENABLED: bool = True
    LOGO_STRIP_MIN_PAGE_FRACTION: float = 0.5
    LOGO_STRIP_MIN_PAGE_FLOOR: int = 3

    # ==================== API KEYS ====================
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    TAVILY_API_KEY: str = ""
    OPENWEATHER_API_KEY: Optional[str] = None      # ← NEW
    GITHUB_TOKEN: Optional[str] = None              # ← NEW
    SLACK_BOT_TOKEN: Optional[str] = None           # ← NEW
    SMTP_HOST: str = "smtp.gmail.com"               # ← NEW
    SMTP_USER: Optional[str] = None                 # ← NEW
    SMTP_PASSWORD: Optional[str] = None             # ← NEW
    SMTP_PORT: int = 587                            # ← NEW
    VOYAGE_API_KEY: Optional[str] = None

    # ==================== SECURITY ====================
    # NOTE: standardized on SECRET_KEY everywhere (app code, .env, .env.example,
    # docker-compose.yml, docker-compose.prod.yml). Previously some compose
    # files set JWT_SECRET_KEY instead, which pydantic-settings does NOT map
    # onto SECRET_KEY -- that caused a required-field crash on container boot.
    SECRET_KEY: str  # Required from .env
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_HOURS: int = 24

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