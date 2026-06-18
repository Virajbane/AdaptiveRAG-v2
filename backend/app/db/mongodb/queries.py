from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId
from datetime import datetime
from app.db.mongodb.models import UserInDB

class UserQueries:
    """User database operations"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db["users"]
    
    async def create_user(self, email: str, name: str, password_hash: str) -> str:
        """Create new user"""
        user = {
            "email": email,
            "name": name,
            "password_hash": password_hash,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "is_active": True,
            "is_verified": False,
            "preferences": {
                "preferred_llm": "ollama",
                "temperature": 0.7,
                "max_tokens": 2000
            }
        }
        result = await self.collection.insert_one(user)
        return str(result.inserted_id)
    
    async def get_user_by_email(self, email: str) -> dict:
        """Get user by email"""
        user = await self.collection.find_one({"email": email})
        return user
    
    async def get_user_by_id(self, user_id: str) -> dict:
        """Get user by ID"""
        user = await self.collection.find_one({"_id": ObjectId(user_id)})
        return user
    
    async def user_exists(self, email: str) -> bool:
        """Check if user already exists"""
        user = await self.collection.find_one({"email": email})
        return user is not None
    
    async def update_user(self, user_id: str, data: dict) -> bool:
        """Update user data"""
        result = await self.collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {**data, "updated_at": datetime.utcnow()}}
        )
        return result.modified_count > 0