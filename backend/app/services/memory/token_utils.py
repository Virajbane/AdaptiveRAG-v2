"""
Conservative token-count estimation for memory/history sizing.

No tokenizer is currently wired in anywhere in this codebase (see
provider.py — Ollama/Groq calls both just send raw prompt strings).
Rather than block this fix on picking and adding a real tokenizer
(tiktoken doesn't match qwen2.5's actual vocab anyway, and the Groq
model's tokenizer differs again), this uses a deliberately conservative
chars-per-token heuristic. It will overestimate for some text and
underestimate for other text, but erring toward triggering
summarization/trimming a bit earlier than strictly necessary is the
safe direction here — the failure mode we're avoiding is silent
overflow, not premature summarization.

Replace with a real tokenizer call once one is chosen (tracked
separately from this fix — see Section 1.1 discussion).
"""

# Conservative average for English prose; dense numeric/table text
# (like this project's table_fig_* golden items) tokenizes more richly
# than this ratio assumes, which is fine — it just means we trigger
# a bit earlier on exactly the content most worth being careful with.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def total_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(m.get("content", "")) for m in messages)