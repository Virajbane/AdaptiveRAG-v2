from app.services.memory.short_term import short_term_memory
from app.services.memory.long_term import LongTermMemory
from typing import List, Optional

class MemoryManager:
    """
    Coordinate short-term and long-term memory
    """
    
    def __init__(self, long_term: LongTermMemory):
        self.short_term = short_term_memory
        self.long_term = long_term
    
    async def load_context(
        self,
        user_id: str,
        session_id: str
    ) -> dict:
        """
        Load all memory context for a user
        
        Returns:
            {
                "history": [...],
                "summaries": [...],
                "preferences": {...}
            }
        """
        
        # Get short-term history
        history = await self.short_term.get_history(user_id, session_id)
        
        # Get long-term summaries
        summaries = await self.long_term.get_recent_summaries(user_id, limit=3)
        
        # Get preferences
        model_pref = await self.long_term.get_preference(user_id, "preferred_model", "qwen2.5:7b")
        
        return {
            "history": history,
            "summaries": summaries,
            "preferences": {
                "preferred_model": model_pref
            }
        }
    
    async def save_interaction(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_response: str
    ) -> bool:
        """Save user and assistant messages"""
        
        # Save to short-term
        await self.short_term.save_message(
            user_id, session_id, "user", user_message
        )
        
        await self.short_term.save_message(
            user_id, session_id, "assistant", assistant_response
        )
        
        return True
    
    async def update_preference(
        self,
        user_id: str,
        key: str,
        value: any
    ) -> bool:
        """Update user preference"""
        return await self.long_term.save_preference(user_id, key, value)

# Will be initialized in main.py
memory_manager = None