from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
from datetime import datetime
from typing import Optional, List

class UserCreate(BaseModel):
    """User registration request"""
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    """User login request"""
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    """User response (without password)"""
    user_id: str = Field(alias="_id")
    email: str
    name: str
    created_at: datetime
    
    class Config:
        populate_by_name = True  # Allow both user_id and _id

class UserInDB(BaseModel):
    """User in database (with password hash)"""
    _id: str
    email: str
    name: str
    password_hash: str
    created_at: datetime
    updated_at: datetime
    is_active: bool = True
    is_verified: bool = False  # Email verification



class DocumentMetadata(BaseModel):
    """Document metadata"""
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    language: str = "en"
    keywords: List[str] = []

class DocumentChunkInfo(BaseModel):
    """Chunk information"""
    count: int
    average_tokens: int
    total_tokens: int
    overlap_tokens: int = 50

class DocumentUpload(BaseModel):
    """Document upload request"""
    filename: str
    # file is handled separately as UploadFile

class DocumentResponse(BaseModel):
    """Document response"""
    doc_id: str = Field(alias="_id")
    user_id: str
    filename: str
    file_type: str  # pdf, docx, txt, csv
    file_size_bytes: int
    status: str  # processing, processed, failed
    chunks: DocumentChunkInfo
    created_at: datetime
    
    class Config:
        populate_by_name = True