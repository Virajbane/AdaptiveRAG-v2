from typing import Any

from app.services.memory.short_term import short_term_memory
from app.services.memory.long_term import LongTermMemory


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

        try:
            # Get short-term history
            history = await self.short_term.get_history(
                user_id,
                session_id
            )

            # Get long-term summaries
            summaries = []
            try:
                summaries = await self.long_term.get_recent_summaries(
                    user_id,
                    limit=3
                )
            except Exception:
                pass

            # Get preferences
            preferences = {}

            try:
                model_pref = await self.long_term.get_preference(
                    user_id,
                    "preferred_model",
                    "qwen2.5:7b"
                )

                preferences = {
                    "preferred_model": model_pref
                }

            except Exception:
                preferences = {
                    "preferred_model": "qwen2.5:7b"
                }

            return {
                "history": history or [],
                "summaries": summaries or [],
                "preferences": preferences
            }

        except Exception as e:
            print(f"Error loading context: {e}")

            return {
                "history": [],
                "summaries": [],
                "preferences": {
                    "preferred_model": "qwen2.5:7b"
                }
            }

    async def save_interaction(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_response: str
    ) -> bool:
        """
        Save user and assistant messages
        """

        await self.short_term.save_message(
            user_id,
            session_id,
            "user",
            user_message
        )

        await self.short_term.save_message(
            user_id,
            session_id,
            "assistant",
            assistant_response
        )

        return True

    async def update_preference(
        self,
        user_id: str,
        key: str,
        value: Any
    ) -> bool:
        """
        Update user preference
        """
        return await self.long_term.save_preference(
            user_id,
            key,
            value
        )


# Will be initialized in main.py
memory_manager = None