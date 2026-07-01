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
  - On any failure (LLM error, empty/unusable response), state.rewritten_question
    stays "" so callers fall back to state.question automatically.
    This follows BaseAgent.run()'s existing convention: _execute()
    catches its own soft failures and never raises for "just use the
    fallback" cases; only unexpected exceptions propagate up to
    BaseAgent.run(), which already sets state.error generically.
"""

import app.services.memory.manager as mm_module
from app.agents.base import BaseAgent
from app.agents.state import AgentState

REWRITE_SYSTEM_PROMPT = """You are a query normalizer for a retrieval system.

Given the recent conversation history and a new user question, rewrite the new \
question as a single standalone, well-formed question.

Rules:
- Resolve pronouns and implicit references using the history (e.g. "what about \
X?" becomes "What is <topic from history> for X?").
- Fix obvious spelling and grammar mistakes ONLY. Fix individual misspelled \
words in place — do not restructure the sentence into a different form.
- NEVER change grammatical person or pronouns. If the user wrote "my", "I", \
or "me", the rewrite MUST keep "my", "I", or "me" exactly — do NOT convert \
to "your", "you", or any other person. This applies even when also fixing \
spelling in the same sentence (e.g. "wha t is my CGPA" must stay "What is \
my CGPA?" — NOT become "What is your CGPA?"). Getting this wrong makes the \
question sound like it's about someone else's data instead of the user's own.
- Do NOT expand acronyms or abbreviations (e.g. "CGPA", "RAG", "API") into \
their full form, and do NOT guess what an acronym stands for. Leave \
acronyms exactly as the user typed them, correcting only obvious case/spacing \
typos if present (e.g. "cgpa" -> "CGPA" is fine; inventing or guessing what \
it expands to is not).
- PRESERVE the original intent/action exactly. If the user gave a command \
("summarize X", "explain X", "compare X and Y"), the rewrite MUST keep that \
same command form. Never collapse a command into a topic label or noun phrase \
(e.g. "summarize the rag2.0 pdf" must stay a summarize-command like "Summarize \
the RAG 2.0 PDF" — NOT become "RAG 2.0 PDF summary").
- Do NOT change the meaning or add information that isn't implied by the \
history or the question.
- Do NOT answer the question.
- Do NOT add commentary, quotes, or explanation.
- If the question is already standalone and clean, return it unchanged \
(just correcting trivial spelling if needed).
- Output ONLY the rewritten question, nothing else.


"""

MAX_HISTORY_TURNS = 3


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

        if rewritten and rewritten != original_question:
            print(f'[REWRITER] "{original_question}" -> "{rewritten}"')

        state.rewritten_question = rewritten
        return state