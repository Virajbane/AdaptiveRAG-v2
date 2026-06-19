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
    
from datetime import datetime
from bson import ObjectId

class DocumentQueries:
    """Document database operations"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db["documents"]
    
    async def create_document(
        self,
        user_id: str,
        filename: str,
        file_type: str,
        file_size_bytes: int,
        storage_path: str
    ) -> str:
        """Create document record"""
        doc = {
            "user_id": user_id,
            "filename": filename,
            "file_type": file_type,
            "file_size_bytes": file_size_bytes,
            "storage_path": storage_path,
            "status": "processing",
            "processing_error": None,
            "chunks": {
                "count": 0,
                "average_tokens": 0,
                "total_tokens": 0,
                "overlap_tokens": 50
            },
            "vectors": {
                "count": 0,
                "stored_in_qdrant": False,
                "namespace": f"user_{user_id}"
            },
            "tags": [],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "usage": {
                "times_retrieved": 0,
                "last_retrieved_at": None
            }
        }
        result = await self.collection.insert_one(doc)
        return str(result.inserted_id)
    
    async def get_document(self, doc_id: str, user_id: str) -> dict:
        """Get document (with user isolation)"""
        doc = await self.collection.find_one({
            "_id": ObjectId(doc_id),
            "user_id": user_id
        })
        return doc
    
    async def list_documents(self, user_id: str, skip: int = 0, limit: int = 20) -> list:
        """List user's documents"""
        docs = await self.collection.find(
            {"user_id": user_id}
        ).sort("created_at", -1).skip(skip).limit(limit).to_list(length=limit)
        return docs
    
    async def update_document_status(
        self,
        doc_id: str,
        status: str,
        error: str = None,
        chunks_info: dict = None
    ) -> bool:
        """Update document status during processing"""
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow()
        }
        
        if error:
            update_data["processing_error"] = error
        
        if chunks_info:
            update_data["chunks"] = chunks_info
            update_data["vectors"] = {
                "count": chunks_info.get("count", 0),
                "stored_in_qdrant": True,
                "namespace": update_data.get("namespace", "")
            }
        
        result = await self.collection.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": update_data}
        )
        return result.modified_count > 0
    
    async def delete_document(self, doc_id: str, user_id: str) -> bool:
        """Delete document"""
        result = await self.collection.delete_one({
            "_id": ObjectId(doc_id),
            "user_id": user_id
        })
        return result.deleted_count > 0