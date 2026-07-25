"""
test_multiturn_memory.py

Minimal harness for the 3 synthetic multi-turn cases. Mocks Redis and
Mongo in-memory so this runs standalone, no real infra needed.
"""

import asyncio
from app.services.memory.short_term import ShortTermMemory
from app.services.memory.manager import MemoryManager
from app.services.llm.provider import LLMProvider


# ---- In-memory fakes, matching the real clients' method signatures ----

class FakeRedis:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, expire=86400):
        self.store[key] = value
        return True

    async def delete(self, key):
        self.store.pop(key, None)
        return True


class FakeLongTerm:
    def __init__(self):
        self.summaries = []

    async def save_session_summary(self, user_id, session_id, summary, topics):
        self.summaries.append({
            "user_id": user_id, "session_id": session_id,
            "summary": summary, "topics": topics,
        })
        return True

    async def get_recent_summaries(self, user_id, limit=5):
        return [s for s in self.summaries if s["user_id"] == user_id][-limit:]

    async def get_preference(self, user_id, key, default=None):
        return default


async def run_case_01(manager):
    """multiturn_01 — normal follow-up resolution still works."""
    print("\n=== multiturn_01: baseline resolution ===")
    await manager.save_interaction(
        "u1", "s1",
        "What UTMOS score does Freeze-Omni achieve?",
        "Freeze-Omni achieves a UTMOS score of 4.21.",
    )
    context = await manager.load_context("u1", "s1")
    print(f"History turns: {len(context['history'])}, Summaries: {len(context['summaries'])}")
    assert len(context["history"]) == 2, "Expected both turns present, well within budget"
    print("PASS: short history preserved, no premature eviction")


async def run_case_02(manager, prose_pairs):
    """multiturn_02 — eviction produces a real summary, not silent loss."""
    print("\n=== multiturn_02: eviction + summarization ===")
    for q, a in prose_pairs:
        await manager.save_interaction("u2", "s2", q, a)

    context = await manager.load_context("u2", "s2")
    print(f"Remaining history: {len(context['history'])}, Summaries saved: {len(context['summaries'])}")

    assert len(context["summaries"]) > 0, "FAIL: no summary was created — data was lost silently"

    # NEW: fail loudly if every summary is actually the crude fallback,
    # not a real LLM summary — this masked a real bug in the first run.
    fell_back = [s for s in context["summaries"] if s["summary"].startswith("(unsummarized excerpt)")]
    assert len(fell_back) == 0, (
        f"FAIL: {len(fell_back)}/{len(context['summaries'])} summaries used the "
        f"crude fallback, not real LLM summarization — check summarizer.py's "
        f"JSON parsing and/or Ollama connectivity"
    )

    all_summary_text = " ".join(s["summary"] for s in context["summaries"])
    assert "StepAudio-2-mini" in all_summary_text or "stepaudio" in all_summary_text.lower(), (
        "FAIL: earliest turn's fact not found in any summary — evicted content wasn't captured"
    )
    print("PASS: evicted turns were summarized via real LLM call and contain the original fact")


async def run_case_03(manager, long_answer_text):
    """multiturn_03 — a single oversized turn is still included, never dropped."""
    print("\n=== multiturn_03: token-budget trim, current turn always kept ===")
    await manager.save_interaction(
        "u3", "s3",
        "Summarize the two architectural innovations in this paper.",
        long_answer_text,
    )
    context = await manager.load_context("u3", "s3")

    from app.agents.rewriter import _select_recent_history, HISTORY_PROMPT_TOKEN_BUDGET
    selected = _select_recent_history(context["history"], HISTORY_PROMPT_TOKEN_BUDGET)

    assert len(selected) >= 1, "FAIL: the most recent turn was dropped entirely"
    print(f"PASS: {len(selected)} turn(s) selected, most recent turn always present")


async def main():
    fake_redis = FakeRedis()
    import app.services.memory.short_term as st_module
    st_module.redis_client = fake_redis  # patch module-level import

    short_term = ShortTermMemory()
    long_term = FakeLongTerm()
    fast_llm = LLMProvider(model="qwen2.5:0.5b")  # adjust if Ollama not running locally
    manager = MemoryManager(long_term=long_term, llm=fast_llm)
    manager.short_term = short_term

    await run_case_01(manager)

    prose_pairs = [
        ("What backbone model does Lychee-FD use as its half-duplex base?",
         "Lychee-FD uses StepAudio-2-mini as its half-duplex backbone."),
        ("What audio encoder does it use?", "Whisper-v3-large."),
        ("What tokenizer converts audio into discrete speech tokens?", "CosyVoice2."),
        ("What frame rate does the acoustic tokenizer use?", "25Hz."),
        ("How many shared Transformer layers make up the backbone?", "24 layers."),
        ("What optimizer is used to train it?", "AdamW."),
        ("What global batch size is used?", "32."),
    ]
    await run_case_02(manager, prose_pairs)

    long_answer = (
        "The paper introduces two innovations: hierarchical parameter separation "
        "and a semantic alignment channel. " * 40  # padded to force >800 est. tokens
    )
    await run_case_03(manager, long_answer)

    print("\nAll cases completed.")


if __name__ == "__main__":
    asyncio.run(main())