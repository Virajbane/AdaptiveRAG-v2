"""
Integration test: does token_estimator's trigger point, combined with
summarizer.py's LLM summarization, actually preserve facts that a
later question would need?

This is the ONLY place token_estimator.py's behavior is observable
end-to-end -- it doesn't touch document retrieval directly, it only
decides *when* short-term history gets compressed. So "does this file
help our RAG" is really "does content evicted under its threshold
still survive, in usable form, in the resulting summary."

Uses facts from golden_set.json (the Lychee-FD paper QA set) as the
"content worth not losing", since that's the corpus already in use
for retrieval eval -- reusing it here keeps both eval suites talking
about the same facts.
"""
import json
import pytest
from pathlib import Path

from app.services.memory.token_utils import estimate_tokens, total_tokens
from app.services.memory.summarizer import summarize_turns


GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


def _load_fact_bearing_turns():
    """
    Build synthetic conversation turns out of a handful of golden-set
    'answer' items, simulating a long chat where the user asked about
    the paper early on, then kept chatting until eviction fired.
    """
    data = json.loads(GOLDEN_SET_PATH.read_text())
    facts = [
        item for item in data["items"]
        if item["type"] == "answer" and not item.get("expects_decline")
    ][:5]

    turns = []
    for f in facts:
        turns.append({"role": "user", "content": f["question"]})
        turns.append({"role": "assistant", "content": f["reference"]})
    return facts, turns


class FakeLLM:
    """Deterministic stand-in so this test doesn't require a live model.
    Swap for the real provider in CI once available; this checks the
    *pipeline wiring*, not model quality."""
    async def generate(self, prompt, max_tokens=200):
        # crude but deterministic "summary": keep first sentence of
        # every assistant turn embedded in the prompt
        import re
        assistant_lines = re.findall(r"assistant: (.+)", prompt)
        summary = " ".join(s.split(".")[0] + "." for s in assistant_lines)
        return json.dumps({"summary": summary[:600], "topics": ["lychee-fd"]})


class TestEvictionThresholdBehavior:
    def test_threshold_fires_before_real_overflow_not_after(self):
        """
        The whole point of a conservative estimator is to trigger
        eviction a bit EARLY. Confirm total_tokens() over-, not
        under-, counts relative to a stricter reference ratio, so we
        never silently blow a real context window.
        """
        _, turns = _load_fact_bearing_turns()
        conservative = total_tokens(turns)
        # a more generous 6-chars/token reference (many real
        # tokenizers land nearer this for English prose)
        generous_chars_per_token = 6
        reference = sum(
            max(1, len(t.get("content", "")) // generous_chars_per_token)
            for t in turns
        )
        assert conservative >= reference, (
            "estimator should over-count (trigger earlier) relative to "
            "a looser reference ratio -- if it under-counts, eviction "
            "fires too late and risks real overflow"
        )


class TestEvictionPreservesFacts:
    @pytest.mark.asyncio
    async def test_evicted_turns_summary_retains_key_values(self):
        """
        Simulate: turns accumulate until total_tokens() crosses a
        budget -> those turns get evicted -> summarize_turns() runs on
        them. Check specific numeric/named facts from the golden set
        (the things fab_* items exist to catch fabrication of) survive
        into the summary text.
        """
        facts, turns = _load_fact_bearing_turns()
        budget = 50  # small budget to force eviction deterministically

        evicted = []
        running = []
        for t in turns:
            running.append(t)
            if total_tokens(running) > budget:
                evicted = running
                break

        assert evicted, "test setup problem: never crossed budget"

        summary, topics = await summarize_turns(evicted, FakeLLM())

        missing = []
        for f in facts:
            # only check facts whose Q/A pair was actually in the
            # evicted slice
            if any(f["question"] in t.get("content", "") for t in evicted):
                key_terms = [w for w in f["reference"].split() if len(w) > 6][:3]
                if not any(term.strip(".,") in summary for term in key_terms):
                    missing.append(f["id"])

        assert not missing, (
            f"summary dropped key terms for evicted facts: {missing}. "
            "This means the eviction threshold + summarizer combo is "
            "losing information a later question could need -- exactly "
            "the failure mode the 2026-07-25 fix in summarizer.py was "
            "meant to prevent."
        )