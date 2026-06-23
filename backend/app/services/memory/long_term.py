from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import Optional, List

class LongTermMemory:
    """
    Long-term memory: User preferences and summaries
    
    Stores:
    - User preferences (model, temperature, etc.)
    - Session summaries
    - Important facts about user
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db["memory_long_term"]
    
    async def save_preference(
        self,
        user_id: str,
        key: str,
        value: any
    ) -> bool:
        """Save user preference"""
        
        try:
            await self.collection.update_one(
                {"user_id": user_id, "type": "preference", "key": key},
                {
                    "$set": {
                        "value": value,
                        "updated_at": datetime.utcnow()
                    },
                    "$setOnInsert": {
                        "created_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
            return True
        except Exception as e:
            print(f"Error saving preference: {e}")
            return False
    
    async def get_preference(
        self,
        user_id: str,
        key: str,
        default: any = None
    ) -> any:
        """Get user preference"""
        
        try:
            doc = await self.collection.find_one({
                "user_id": user_id,
                "type": "preference",
                "key": key
            })
            return doc["value"] if doc else default
        except Exception as e:
            print(f"Error getting preference: {e}")
            return default
    
    async def save_session_summary(
        self,
        user_id: str,
        session_id: str,
        summary: str,
        topics: List[str]
    ) -> bool:
        """Save session summary"""
        
        try:
            await self.collection.insert_one({
                "user_id": user_id,
                "type": "session_summary",
                "session_id": session_id,
                "summary": summary,
                "topics": topics,
                "created_at": datetime.utcnow()
            })
            return True
        except Exception as e:
            print(f"Error saving session summary: {e}")
            return False
    
    async def get_recent_summaries(
        self,
        user_id: str,
        limit: int = 5
    ) -> List[dict]:
        """Get recent session summaries"""
        
        try:
            summaries = await self.collection.find({
                "user_id": user_id,
                "type": "session_summary"
            }).sort("created_at", -1).limit(limit).to_list(length=limit)
            
            # Convert ObjectId and datetime to JSON-safe types
            clean_summaries = []
            for s in summaries:
                clean_summaries.append({
                    "id": str(s["_id"]),
                    "session_id": s.get("session_id"),
                    "summary": s.get("summary"),
                    "topics": s.get("topics", []),
                    "created_at": s["created_at"].isoformat() if s.get("created_at") else None
                })
            
            return clean_summaries
        except Exception as e:
            print(f"Error getting summaries: {e}")
            return []

# Will be initialized in main.py
long_term_memory = None