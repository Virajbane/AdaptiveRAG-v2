from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime

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