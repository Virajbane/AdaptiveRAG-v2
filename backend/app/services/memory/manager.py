from typing import Any

from app.services.memory.short_term import short_term_memory
from app.services.memory.long_term import LongTermMemory
from app.services.memory.summarizer import summarize_turns
from app.services.llm.provider import LLMProvider
from app.config.settings import settings


class MemoryManager:
    """
    Coordinate short-term and long-term memory
    """

    def __init__(self, long_term: LongTermMemory, llm: LLMProvider = None):
        self.short_term = short_term_memory
        self.long_term = long_term
        # 2026-07-25: needed to summarize turns evicted from short-term
        # storage before they're lost — see save_interaction below.
        # Defaults to the fast/routing model (this is a cheap,
        # low-stakes compression step, not the user-facing answer path)
        # so existing call sites constructing MemoryManager(long_term)
        # continue to work unchanged.
        self.llm = llm or LLMProvider(model=settings.OLLAMA_FAST_MODEL)

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

    async def _on_evict(self, user_id: str, session_id: str, evicted: list[dict]):
        """
        Callback passed to ShortTermMemory.save_message. Summarizes
        turns being dropped from short-term storage and persists the
        summary to long-term memory, so nothing is lost outright — see
        2026-07-25 fix note in short_term.py.
        """
        summary, topics = await summarize_turns(evicted, self.llm)
        if summary:
            await self.long_term.save_session_summary(
                user_id, session_id, summary, topics
            )

    async def save_interaction(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_response: str
    ) -> bool:
        """
        Save user and assistant messages. Any short-term history
        evicted as a result (by count or token budget) is summarized
        into long-term memory first — see _on_evict.
        """

        async def _evict_cb(evicted: list[dict]):
            await self._on_evict(user_id, session_id, evicted)

        await self.short_term.save_message(
            user_id,
            session_id,
            "user",
            user_message,
            on_evict=_evict_cb,
        )

        await self.short_term.save_message(
            user_id,
            session_id,
            "assistant",
            assistant_response,
            on_evict=_evict_cb,
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