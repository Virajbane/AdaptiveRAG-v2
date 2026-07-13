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
            # started_at is set separately by mark_processing_started(),
            # right when the background task actually begins work — not
            # here at record-creation time. If there's ever a queueing
            # delay between "record created" and "task actually started"
            # (e.g. under load, or with a real queue in future), this
            # keeps started_at meaning what it says: when work began, not
            # when the record was inserted. created_at already covers
            # "when was this uploaded."
            "started_at": None,
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
    
    async def mark_processing_started(self, doc_id: str) -> bool:
        """
        Stamp started_at the moment background processing actually begins
        (called at the top of process_document(), before the real work).

        This is the field stale-job detection keys off of — a document
        sitting at status="processing" with a started_at from 20 minutes
        ago, when processing normally takes seconds, is almost certainly
        a job whose background task died (process crash/restart) rather
        than one that's just slow.
        """
        result = await self.collection.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {"started_at": datetime.utcnow(), "updated_at": datetime.utcnow()}}
        )
        return result.modified_count > 0

    async def update_document_status(
        self,
        doc_id: str,
        status: str,
        error: str = None,
        chunks_info: dict = None,
        chunks_failed: int = None,
        failed_chunk_indices: list = None,
        docling_page_errors: list = None,   # NEW
        user_id: str = None,
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
                "count": chunks_info.get("stored_count", chunks_info.get("count", 0)),
                "stored_in_qdrant": True,
                "namespace": f"user_{user_id}" if user_id else "",
            }

        if chunks_failed is not None:
            update_data["chunks_failed"] = chunks_failed

        if failed_chunk_indices is not None:
            update_data["failed_chunk_indices"] = failed_chunk_indices

        if docling_page_errors is not None:   # NEW -- same is-not-None guard
                                                # pattern as the other optional
                                                # fields, so existing callers
                                                # that don't pass it are unaffected
            update_data["docling_page_errors"] = docling_page_errors

        result = await self.collection.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": update_data}
        )
        return result.modified_count > 0

    async def get_stale_processing_documents(self, user_id: str, timeout_minutes: int = 10) -> list:
        """
        Find documents stuck at status="processing" whose started_at is
        older than `timeout_minutes`. These are almost certainly jobs
        whose background task died mid-flight (process restart/crash) —
        a real in-progress job on this codebase should finish well within
        a few minutes based on observed per-node timings.

        Used by the /documents/{doc_id}/retry endpoint and can also back
        a "stuck uploads" indicator in the frontend.
        """
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(minutes=timeout_minutes)
        docs = await self.collection.find({
            "user_id": user_id,
            "status": "processing",
            "started_at": {"$ne": None, "$lt": cutoff},
        }).to_list(length=100)
        return docs

    async def delete_document(self, doc_id: str, user_id: str) -> bool:
        """Delete document"""
        result = await self.collection.delete_one({
            "_id": ObjectId(doc_id),
            "user_id": user_id
        })
        return result.deleted_count > 0