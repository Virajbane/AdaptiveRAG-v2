"""
Summarizes evicted short-term messages into a compact long-term entry.

Deliberately minimal prompt, run on fast_llm — this is a low-stakes,
best-effort compression step, not a user-facing generation. A failure
here should never block message storage (see short_term.py's on_evict
handling), and a weak/off-topic summary is a much smaller regression
than the current behavior of losing the content outright.

2026-07-25 fix: initial version used a raw json.loads(response) with
no markdown-fence handling or balanced-brace extraction. This failed
against the actual local model (qwen2.5:0.5b), which wraps JSON output
in ```json fences — a known behavior already handled elsewhere in this
codebase (BaseAgent.parse_json_response, CriticAgent._extract_json).
Confirmed in testing: every real summarization call fell through to
the crude fallback, silently, because the fallback text still happened
to satisfy the test's assertion — meaning the LLM summarization path
was never actually exercised despite the test reporting PASS. Fixed by
reusing the same balanced-brace JSON extraction already proven to work
against this model elsewhere in the project, instead of re-implementing
a weaker version here.
"""

import json
import re

SUMMARIZE_PROMPT = """Summarize this conversation excerpt in 2-3 sentences, \
preserving specific facts, names, and numbers mentioned. Then list up to 5 \
short topic keywords.

Conversation:
{turns}

Output ONLY JSON in this exact shape:
{{"summary": "...", "topics": ["...", "..."]}}
"""


def _format_turns(turns: list[dict]) -> str:
    return "\n".join(f"{t.get('role', 'user')}: {t.get('content', '')}" for t in turns)


def _extract_balanced_json(text: str) -> str | None:
    """
    Same brace-depth-tracking approach as BaseAgent._extract_balanced_json
    — duplicated here rather than imported, since summarizer.py operates
    on a raw LLMProvider, not a BaseAgent subclass, and this logic is
    small enough not to warrant a shared-utility refactor as part of
    this fix. Finds the first complete top-level {...} object, correctly
    stopping at its end even if a second JSON-looking block follows.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        char = text[i]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_summary_json(response: str) -> dict:
    """
    Mirrors BaseAgent.parse_json_response's fallback chain: try direct
    parse, then strip markdown fences, then balanced-brace scan. Raises
    if none work, so the caller's except block can trigger the
    crude-fallback path deliberately, rather than this function
    swallowing the failure itself.
    """
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', response.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    candidate = _extract_balanced_json(response)
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Could not extract valid JSON", response, 0)


async def summarize_turns(turns: list[dict], llm) -> tuple[str, list[str]]:
    if not turns:
        return "", []

    prompt = SUMMARIZE_PROMPT.format(turns=_format_turns(turns))

    try:
        response = await llm.generate(prompt, max_tokens=200)
        parsed = _parse_summary_json(response.strip())
        summary = parsed.get("summary", "") or ""
        topics = parsed.get("topics", []) or []
        return summary, topics
    except Exception as e:
        # Fallback: a crude non-LLM summary is still strictly better
        # than losing the content with zero trace of it ever existing.
        print(f"[SUMMARIZER] LLM summarization failed, using fallback: {e}")
        fallback = "; ".join(
            f"{t.get('role')}: {t.get('content', '')[:100]}" for t in turns
        )
        return f"(unsummarized excerpt) {fallback}", []