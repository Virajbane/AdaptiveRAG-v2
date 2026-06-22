from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional, Any

class MessageModel(BaseModel):
    """Chat message"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None

class SessionMemoryRequest(BaseModel):
    """Request to save session memory"""
    session_id: str
    summary: str
    topics: List[str]

class PreferenceModel(BaseModel):
    """User preference"""
    key: str
    value: Any

class MemoryResponse(BaseModel):
    """Memory API response"""
    history: List[MessageModel]
    preferences: dict
    recent_summaries: List[dict]