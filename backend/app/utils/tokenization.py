"""
Shared, provider-aware token counting.

Single source of truth for "how many tokens is this text" across the
codebase -- used by TextChunker, DoclingChunker, and AnswerAgent's
context-window budget check. Previously each of those either used the
wrong tokenizer (tiktoken/cl100k_base -- OpenAI's, matching neither
backend this project runs) or didn't count at all.

Provider split (see app/services/llm/provider.py's own use_groq switch,
which this mirrors):
  - Ollama/local dev (qwen2.5): real tokenizer, loaded via transformers.
    Qwen2.5's tokenizer is ungated on HuggingFace -- free to load, no
    auth needed, and shared across all Qwen2.5 sizes (0.5b, 7b, ...).
  - Groq/production (llama-3.3-70b-versatile): meta-llama's tokenizer
    is GATED on HuggingFace (requires an authenticated account that
    has accepted Meta's license, plus an HF_TOKEN in the container).
    Adding that as a hard dependency for token *counting* on a
    free-tier deploy is exactly the fragility
    app/services/memory/token_utils.py already deliberately avoided
    for the same underlying reason -- so this path reuses that same
    conservative chars-per-token estimate instead.

The Qwen tokenizer is loaded LAZILY and cached at module level (a
process-wide singleton), not per-caller -- DocumentProcessor (and
therefore TextChunker) is constructed fresh per background upload task
(see app/api/v1/endpoints/documents.py's process_document), so without
this, every single document upload would re-load the tokenizer from
disk. One load per process, reused by every caller after that.

2026-08-02 fix — offline fallback:
  First real run surfaced this: AutoTokenizer.from_pretrained() needs
  to reach huggingface.co on its FIRST call (to check/download the
  tokenizer files), and if that network isn't available (dev machine
  with no route out, firewalled server, etc.), the original code let
  that exception propagate straight up -- which crashed the entire
  AnswerAgent, not just token counting. A token-counting convenience
  should never be able to take down answer generation entirely.

  Fixed: wrapped the load in try/except. On failure, permanently fall
  back to the same conservative chars-per-token estimate used on the
  Groq path for the rest of this process's lifetime, and print a clear
  one-time warning explaining why (so it's visible, but doesn't spam
  logs on every single call). This means local/dev runs get exact Qwen
  counts once the tokenizer is cached, and a safe degraded estimate if
  it can never be downloaded -- never a crash either way.
"""

from typing import Optional
from app.config.settings import settings

_CHARS_PER_TOKEN = 4
_QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"

_tokenizer = None       # process-wide singleton, populated on first use
_tokenizer_failed = False  # set permanently if the download ever fails


def get_qwen_tokenizer():
    """Public accessor for callers that need encode/decode directly (e.g.
    hard-splitting an oversized piece of text by raw token slices), not
    just a count. Same cached singleton as count_tokens() uses.

    Raises if the tokenizer can't be loaded -- callers that can tolerate
    a fallback should go through count_tokens() instead, which catches
    this and degrades gracefully."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(_QWEN_MODEL)
    return _tokenizer


def using_groq(use_groq: Optional[bool] = None) -> bool:
    """Resolves the same way LLMProvider does, unless explicitly overridden
    (tests/callers can force either path without touching env vars)."""
    return bool(settings.GROQ_API_KEY) if use_groq is None else use_groq


def count_tokens(text: str, use_groq: Optional[bool] = None) -> int:
    global _tokenizer_failed
    if not text:
        return 0

    use_estimate = using_groq(use_groq) or _tokenizer_failed
    if not use_estimate:
        try:
            return len(get_qwen_tokenizer().encode(text, add_special_tokens=False))
        except Exception as e:
            # No route to huggingface.co (or corrupted/missing local
            # cache) -- degrade permanently for this process rather than
            # retrying the network on every single call, and never let
            # this exception reach the caller.
            _tokenizer_failed = True
            print(
                f"[TOKENIZATION] Could not load Qwen tokenizer "
                f"({type(e).__name__}: {e}). Falling back to the "
                f"chars-per-token estimate for the rest of this "
                f"process. To get exact counts, ensure this machine "
                f"can reach huggingface.co once (to cache the "
                f"tokenizer files), or pre-download it -- see README/"
                f"deployment notes."
            )

    # Conservative estimate -- see module docstring. Deliberately not
    # exact; erring toward slightly smaller chunks/budget than a real
    # tokenizer would produce is the safe direction (undercounting
    # risks overflow, overcounting only costs a little headroom).
    return max(1, len(text) // _CHARS_PER_TOKEN)