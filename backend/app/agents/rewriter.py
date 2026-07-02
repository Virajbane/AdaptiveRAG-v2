"""
backend/app/agents/rewriter.py

RewriterAgent: runs first in the graph, before the planner/retriever
fan-out. Combines two jobs in one fast-model call:

  1. Context resolution — turn a follow-up like "what about digital
     products?" into a standalone question, using short-term memory.
  2. Normalization — fix obvious typos/spelling/grammar so retrieval
     (BM25 + dense) and the Planner's routing classifier both see a
     clean query.

Design notes:
  - Always runs (no heuristic skip-gate). A naive spellcheck heuristic
    would misfire on domain terms like "Qdrant" or "RAG", so we let
    the fast model (qwen2.5 fast/routing model) handle both jobs in
    one pass every turn. It's cheap relative to the main 7B calls.
  - state.question is NEVER mutated — stays as the user's literal
    input, for transcript/UI display and for save_interaction().
  - state.rewritten_question is the new field. Empty string ("") means
    "not rewritten" — downstream nodes (planner, retriever) must use
    `state.rewritten_question or state.question`.
  - On any failure (LLM error, empty/unusable response, a rewrite
    that drops a personal-reference word present in the original, or
    a rewrite that diverges too far from the original's actual
    content), state.rewritten_question stays "" so callers fall back
    to state.question automatically. This follows BaseAgent.run()'s
    existing convention: _execute() catches its own soft failures and
    never raises for "just use the fallback" cases; only unexpected
    exceptions propagate up to BaseAgent.run(), which already sets
    state.error generically.

2026-06-30 bug note:
  - The fast model was observed rewriting "What are my skills mentioned
    in the docs?" into "What specific skills are listed in the
    documentation?" — dropping "my" entirely (not converting it to
    "your", which the prompt already banned) and swapping "docs" for a
    synonym. This silently strips the signal the Planner uses to route
    to ["documents"], causing misrouting to ["web"]. Prompt rules were
    tightened to explicitly ban deletion/synonym-substitution (not just
    person-swapping), and a code-level safety net
    (_dropped_personal_reference) was added as a backstop independent
    of prompt compliance, since this is a fast model and prompt-only
    guarantees weren't reliable enough for something this consequential
    downstream.

2026-07-02 bug note:
  - Separately, the fast model was observed rewriting "Who won the most
    recent Formula 1 race?" into "Who is the most successful driver
    among the top five drivers of all time?" — not a spelling fix or a
    personal-reference drop, but a full hallucinated substitute
    question on the same general topic. _dropped_personal_reference
    doesn't catch this (no personal reference involved at all), so a
    second, broader backstop (_diverged_too_much) was added: it flags
    rewrites whose overall text is no longer substantially similar to
    the original, regardless of *which* words changed. This is
    deliberately generic rather than another keyword-specific check,
    since the underlying problem (small model not reliably following
    "do not restructure the sentence") isn't limited to any one phrase
    pattern and hand-coding each observed case doesn't scale.
"""

import re
from difflib import SequenceMatcher

import app.services.memory.manager as mm_module
from app.agents.base import BaseAgent
from app.agents.state import AgentState

REWRITE_SYSTEM_PROMPT = """Rewrite the question as one standalone, well-formed question.

Rules:
- Resolve pronouns/references using history (e.g. "what about X?" -> full topic + X).
- Fix spelling only. Do not restructure the sentence.
- Keep every word EXACTLY as written. Never delete or swap in a synonym for \
"my"/"I"/"me" or for the user's own words like "docs"/"report"/"notes" \
("docs" must NOT become "documentation").
- Never invent, substitute, or answer a different question. If you are unsure \
how to fix something, leave it unchanged rather than guessing a replacement.
- Never expand acronyms (CGPA, RAG, API stay as-is). Never guess what one means.
- Keep the same command form (e.g. "summarize X" stays "summarize X", not "X summary").
- Output ONLY the rewritten question. No explanation.

Example:
Input: "wha t is my CGPA"
Output: "What is my CGPA?"

Example:
Input: "summarixe the rag2.0 pdf"
Output: "Summarize the RAG 2.0 PDF"

Example:
Input: "What are my skills mentioned in the docs?"
Output: "What are my skills mentioned in the docs?"

Example (do NOT do this — this invents a different question):
Input: "Who won the most recent Formula 1 race?"
WRONG Output: "Who is the most successful driver among the top five drivers of all time?"
RIGHT Output: "Who won the most recent Formula 1 race?"
"""

MAX_HISTORY_TURNS = 3

_PERSONAL_REF = re.compile(r"\b(my|i|me)\b", re.IGNORECASE)

# Below this overall similarity ratio, a rewrite is treated as having
# replaced the question's actual content rather than just cleaning it
# up. Legitimate spelling/grammar fixes and reasonable context
# resolution (expanding "what about X?" using history) both keep this
# fairly high, since most of the original characters survive; a
# hallucinated substitute question does not. Tune based on observed
# false positives/negatives in logs.
_DIVERGENCE_SIMILARITY_THRESHOLD = 0.4


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(no prior conversation)"
    lines = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _looks_unusable(text: str) -> bool:
    if not text:
        return True
    cleaned = text.strip()
    if len(cleaned) < 3:
        return True
    if cleaned.lower().startswith(("rules:", "system:", "i cannot", "i can't")):
        return True
    return False


def _dropped_personal_reference(original: str, rewritten: str) -> bool:
    """
    Safety net independent of prompt compliance.

    If the user's original question contains "my"/"I"/"me" and the
    rewritten version doesn't contain any of them, the rewrite has
    stripped a signal the Planner depends on for routing to
    ["documents"] — even if the model didn't do the more obvious
    "your"/"you" swap the prompt already bans. Deliberately narrow: it
    only checks for this one known failure mode, not general rewrite
    drift, to avoid false-positiving on legitimate rewrites.
    """
    return bool(_PERSONAL_REF.search(original)) and not _PERSONAL_REF.search(rewritten)


def _diverged_too_much(original: str, rewritten: str) -> bool:
    """
    Broader safety net, complementary to _dropped_personal_reference.

    Catches cases where the model replaces the question's actual
    content with a different (but topically related) question, rather
    than just fixing spelling/grammar or resolving a reference — e.g.
    "who won the most recent race?" -> "who is the most successful
    driver of all time?". Both questions are "about F1", but they ask
    for different facts, and downstream retrieval/planning need the
    real one.

    Uses overall string similarity rather than matching specific
    words/phrases, since the failure isn't limited to one vocabulary
    pattern (unlike the personal-reference case) — any part of the
    question can get silently swapped out. This is a heuristic, not a
    semantic check, so it's intentionally conservative (relies on the
    fact that legitimate fixes barely touch the original text) rather
    than trying to judge meaning.
    """
    ratio = SequenceMatcher(None, original.lower(), rewritten.lower()).ratio()
    return ratio < _DIVERGENCE_SIMILARITY_THRESHOLD


class RewriterAgent(BaseAgent):
    """
    Resolves conversational context and normalizes spelling/grammar
    before planning/retrieval. See module docstring for design notes.
    """

    async def _execute(self, state: AgentState) -> AgentState:
        original_question = state.question

        # ---- Fetch short-term history ----
        history = []
        if mm_module.memory_manager:
            try:
                history = await mm_module.memory_manager.short_term.get_history(
                    state.user_id, state.session_id
                )
            except Exception as e:
                print(f"[REWRITER] Failed to fetch history, proceeding without it: {e}")
                history = []
        else:
            print("[REWRITER] memory_manager not initialized, proceeding without history")

        history_text = _format_history(history)

        prompt = (
            f"{REWRITE_SYSTEM_PROMPT}\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"New question: \"{original_question}\"\n\n"
            f"Rewritten standalone question:"
        )

        # ---- Call fast LLM ----
        try:
            response = await self.call_llm(prompt)
            rewritten = response.strip().strip('"')
        except Exception as e:
            print(f"[REWRITER] LLM call failed, falling back to original question: {e}")
            rewritten = ""

        if _looks_unusable(rewritten):
            print("[REWRITER] Rewrite unusable, falling back to original question")
            rewritten = ""

        if rewritten and _dropped_personal_reference(original_question, rewritten):
            print(
                f'[REWRITER] Rewrite dropped personal reference ("my"/"I"/"me"), '
                f'falling back to original question. Bad rewrite was: "{rewritten}"'
            )
            rewritten = ""

        # Only meaningful to check divergence against the raw question,
        # not against history-resolved context — a follow-up like
        # "what about X?" is SUPPOSED to look very different from its
        # resolved form. history presence isn't tracked in state here,
        # but in practice a no-history rewrite that diverges this much
        # is virtually always a hallucinated replacement rather than
        # legitimate resolution, since there's nothing to resolve from.
        if rewritten and not history and _diverged_too_much(original_question, rewritten):
            print(
                f'[REWRITER] Rewrite diverged too far from original (similarity below '
                f'{_DIVERGENCE_SIMILARITY_THRESHOLD}), falling back to original question. '
                f'Bad rewrite was: "{rewritten}"'
            )
            rewritten = ""

        if rewritten and rewritten != original_question:
            print(f'[REWRITER] "{original_question}" -> "{rewritten}"')

        state.rewritten_question = rewritten
        return state