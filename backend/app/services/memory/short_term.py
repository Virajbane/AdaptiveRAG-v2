import json
from datetime import datetime
from typing import List
from app.services.memory.redis_client import redis_client

class ShortTermMemory:
    """
    Short-term memory: Recent conversation history
    
    Stores last 10 messages per session in Redis
    Expires after 24 hours
    """
    
    def __init__(self):
        self.max_messages = 10
        self.ttl = 86400  # 24 hours
    
    def _session_key(self, user_id: str, session_id: str) -> str:
        """Build Redis key for session"""
        return f"session:{user_id}:{session_id}"
    
    async def save_message(
        self,
        user_id: str,
        session_id: str,
        role: str,  # "user" or "assistant"
        content: str
    ) -> bool:
        """Save message to session history"""
        
        key = self._session_key(user_id, session_id)
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Get existing messages
            existing = await redis_client.get(key)
            messages = json.loads(existing) if existing else []
            
            # Add new message
            messages.append(message)
            
            # Keep only last 10
            if len(messages) > self.max_messages:
                messages = messages[-self.max_messages:]
            
            # Save back to Redis
            await redis_client.set(
                key,
                json.dumps(messages),
                expire=self.ttl
            )
            
            return True
        except Exception as e:
            print(f"Error saving message: {e}")
            return False
    
    async def get_history(
        self,
        user_id: str,
        session_id: str
    ) -> List[dict]:
        """Get conversation history"""
        
        key = self._session_key(user_id, session_id)
        
        try:
            data = await redis_client.get(key)
            if data:
                return json.loads(data)
            return []
        except Exception as e:
            print(f"Error getting history: {e}")
            return []
    
    async def clear_session(
        self,
        user_id: str,
        session_id: str
    ) -> bool:
        """Clear session history"""
        
        key = self._session_key(user_id, session_id)
        return await redis_client.delete(key)

# Global instance
short_term_memory = ShortTermMemory()