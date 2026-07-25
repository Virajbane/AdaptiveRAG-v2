import json
from datetime import datetime
from typing import List, Optional, Awaitable, Callable

from app.services.memory.redis_client import redis_client
from app.services.memory.token_utils import total_tokens

# 2026-07-25 fix: save_message previously trimmed to the last
# max_messages entries and wrote that back to Redis in the same
# operation — anything sliced off was gone permanently, with no
# summarization step ever having a chance to run against it (the
# long-term summary path existed in long_term.py but nothing evicted
# by short_term ever reached it). This silently lost real conversation
# content once a session passed 5 exchanges (10 messages), independent
# of whether any individual message was actually long.
#
# Fix: eviction now happens by BOTH count (existing behavior, a hard
# ceiling on Redis payload size) and estimated token budget (new — the
# thing RewriterAgent's prompt actually cares about, and can trigger
# eviction earlier than the count cap if a turn is unusually long).
# Either way, anything evicted is handed to an on_evict callback BEFORE
# being dropped, so callers can persist a summary instead of losing it.
# on_evict is optional and best-effort: if it's not provided, or it
# raises, eviction still proceeds — summarization must never block
# storage from working.


class ShortTermMemory:
    """
    Short-term memory: Recent conversation history

    Stores messages per session in Redis, evicting the oldest entries
    once EITHER the message-count cap or the estimated token budget is
    exceeded. Expires after 24 hours.
    """

    def __init__(
        self,
        max_messages: int = 10,
        ttl: int = 86400,
        # ~1200 tokens is a deliberately conservative slice of
        # fast_llm's 2048-token default context — leaves headroom for
        # REWRITE_SYSTEM_PROMPT's own ~400-token overhead plus the
        # current question, without needing this module to know
        # anything about which prompt eventually consumes this history.
        token_budget: int = 1200,
    ):
        self.max_messages = max_messages
        self.ttl = ttl
        self.token_budget = token_budget

    def _session_key(self, user_id: str, session_id: str) -> str:
        """Build Redis key for session"""
        return f"session:{user_id}:{session_id}"

    async def save_message(
        self,
        user_id: str,
        session_id: str,
        role: str,  # "user" or "assistant"
        content: str,
        on_evict: Optional[Callable[[List[dict]], Awaitable[None]]] = None,
    ) -> bool:
        """
        Save message to session history.

        on_evict, if provided, is awaited with the list of messages
        being evicted (oldest-first) BEFORE they're dropped from
        storage — intended for the caller (MemoryManager) to summarize
        them into long-term memory rather than losing them silently.
        """

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

            # Evict oldest-first while EITHER cap is exceeded. Always
            # keep at least 1 message — this should never fully empty
            # a session's history over a single save.
            evicted: List[dict] = []
            while len(messages) > 1 and (
                len(messages) > self.max_messages
                or total_tokens(messages) > self.token_budget
            ):
                evicted.append(messages.pop(0))

            if evicted and on_evict:
                try:
                    await on_evict(evicted)
                except Exception as e:
                    print(
                        f"[SHORT_TERM] on_evict callback failed — "
                        f"{len(evicted)} evicted message(s) will be lost "
                        f"without a long-term summary: {e}"
                    )

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