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
    that drops a personal-reference word, a numeral/quantifier, or an
    acronym present in the original, a rewrite that invents a
    technology/database type not present in the original, a rewrite
    that diverges too far from the original's actual content, or a
    rewrite that resembles a PRIOR turn in history more than it
    resembles the current question), state.rewritten_question stays ""
    so callers fall back to state.question automatically. This follows
    BaseAgent.run()'s existing convention: _execute() catches its own
    soft failures and never raises for "just use the fallback" cases;
    only unexpected exceptions propagate up to BaseAgent.run(), which
    already sets state.error generically.

2026-06-30 bug note:
  - The fast model was observed rewriting "What are my skills mentioned
    in the docs?" into "What specific skills are listed in the
    documentation?" — dropping "my" entirely and swapping "docs" for a
    synonym. This silently strips the signal the Planner uses to route
    to ["documents"], causing misrouting to ["web"]. Prompt rules were
    tightened, and a code-level safety net (_dropped_personal_reference)
    was added as a backstop independent of prompt compliance.

2026-07-02 bug note:
  - Separately, the fast model was observed rewriting "Who won the most
    recent Formula 1 race?" into "Who is the most successful driver
    among the top five drivers of all time?" — a full hallucinated
    substitute question on the same general topic.
    _dropped_personal_reference doesn't catch this, so a second,
    broader backstop (_diverged_too_much) was added: it flags rewrites
    whose overall text is no longer substantially similar to the
    original, regardless of *which* words changed.

  - Originally, _diverged_too_much only ran when `not history`. This
    assumption broke down during multi-turn eval sessions (see below).

2026-07-04 bug note (first pass):
  - During a sequential 20-question eval run, the fast model rewrote
    "According to the paper, what are the estimated annual losses from
    insurance fraud in the US and UK?" into a near-verbatim COPY of the
    previous assistant ANSWER (the paper's title/authors/affiliations
    block). Because `not history` gated _diverged_too_much off, the
    check never ran. Fixed by making _diverged_too_much run
    unconditionally, and by adding _matches_prior_question, which
    compared the rewrite against each prior USER turn via character
    similarity (SequenceMatcher).

  - Separately, "What does FNOL stand for?" was rewritten to "What is
    the abbreviation for 'federal net of losses'?" — violating the
    prompt's "never expand acronyms" rule and corrupting retrieval.
    No code-level backstop existed for this yet, only the prompt rule.

2026-07-04 bug note (second pass — this revision):
  - The character-similarity approach in _matches_prior_question turned
    out to be the wrong tool. In a later eval run, the rewriter
    paraphrased the PRIOR question rather than copying it verbatim:
    prior turn "Which two organizations are the authors affiliated
    with?" became rewrite "According to the paper, which two
    organizations are mentioned as affiliated with the authors?" for a
    question that should have been about annual fraud losses. This is
    semantically the same failure as the original bug (regurgitating
    the wrong turn), but reordering + added preamble drops
    SequenceMatcher's ratio well below the 0.85 threshold, so the check
    didn't fire. Worse, the very first occurrence of this failure class
    was copying a prior ASSISTANT answer, not a user question at all —
    and the old check only ever looked at user turns, so that case was
    never covered by design, not just by threshold miscalibration.

    Replaced _matches_prior_question with _matches_prior_turn, which:
      1. Checks BOTH user and assistant turns in history (a hallucinated
         rewrite can regurgitate either).
      2. Uses word-set overlap (order-independent, paraphrase-tolerant)
         instead of character sequence similarity.
      3. Compares RELATIVELY: does this rewrite share more content with
         a prior turn than it shares with the CURRENT question it's
         supposed to be a rewrite of? A legitimate rewrite — even one
         that pulls entities from history to resolve a follow-up —
         should still resemble the current question at least as much
         as any single prior turn, since it's derived from the current
         question's own content plus resolved references. A
         hallucinated substitute fails that comparison because it isn't
         derived from the current question at all.

    Also added, same session:
      - _dropped_quantifier: catches silent numeral/quantifier loss
        (e.g. "two techniques" -> "the technique"), observed as a
        subtler drift pattern than outright hallucination — not
        gross enough to trip _diverged_too_much, but still corrupting
        the question's actual claim.
      - _dropped_or_altered_acronym: a code-level backstop for the
        FNOL-expansion failure above, which previously had only the
        prompt rule ("never expand acronyms") and no deterministic
        check — consistent with the project's existing pattern
        (_dropped_personal_reference) of not trusting prompt-only
        compliance from the fast model for anything consequential
        downstream.

2026-07-25 fix (context-window overflow + orphaned summaries):
  - History was previously fetched via short_term.get_history() and
    hard-capped to the last MAX_HISTORY_TURNS (3) turns, with no check
    on how many tokens those turns actually cost. If any one of those
    3 turns was long (e.g. a detailed multi-fact AnswerAgent response),
    the combined prompt (REWRITE_SYSTEM_PROMPT + history + question)
    could silently exceed fast_llm's context window, with Ollama
    truncating the prompt with no error and no signal anywhere in this
    file. There was also no eval coverage of this at all — the golden
    dataset has zero multi-turn conversations.
  - Separately, MemoryManager.load_context() (which merges short-term
    history with long-term session summaries) was never called by this
    file — history was fetched directly from short_term, bypassing
    summaries entirely. Combined with ShortTermMemory's hard 10-message
    cap, this meant older conversation content was being permanently
    dropped with no summary ever reaching this prompt, even though
    LongTermMemory already had a working (but unused) summary-storage
    method.
  - Fix: switched to load_context() so summaries are actually included,
    and replaced the fixed 3-turn cap with a token-budgeted selection
    (_select_recent_history) that includes as many recent turns as fit
    inside HISTORY_PROMPT_TOKEN_BUDGET, oldest-dropped-first. This
    caps prompt size regardless of how long any individual turn is.
    _matches_prior_turn below still checks against the FULL history
    list (not just the token-trimmed subset used in the prompt itself),
    since that safety check should compare against everything actually
    stored, not just what fit in this particular prompt.

2026-08-11 fix (Test 4 root cause -- production log investigation):
  - "What is the total number of records in our database?" was rewritten
    to "How many records are there in your SQL database?" -- inventing
    "SQL" (a database TYPE the user never specified) and silently
    swapping "our" -> "your". _dropped_personal_reference existed as a
    backstop for exactly this class of corruption, but its regex
    (_PERSONAL_REF) only matched "my"/"i"/"me" -- it never included
    "our", so a rewrite that drops/alters "our" sailed through
    undetected. Separately, there was no backstop at all against
    inventing a specific database/tech TYPE (SQL, MongoDB, etc.) that
    wasn't in the original question -- only the prompt's generic
    "don't invent things" instruction, which the fast model didn't
    reliably follow. Fixed both: added "our" to _PERSONAL_REF, and added
    a new _invented_tech_type backstop, following the exact same
    pattern as _dropped_or_altered_acronym (code-level check
    independent of prompt compliance, since this project has already
    established that pattern-in this file for anything consequential
    to downstream routing).
"""

import re

import app.services.memory.manager as mm_module
from app.agents.base import BaseAgent
from app.agents.state import AgentState
from app.services.memory.token_utils import estimate_tokens

REWRITE_SYSTEM_PROMPT = """Rewrite the question as one standalone, well-formed question.

Rules:
- Resolve pronouns/references using history (e.g. "what about X?" -> full topic + X).
- Fix spelling only. Do not restructure the sentence.
- Keep every word EXACTLY as written. Never delete or swap in a synonym for \
"my"/"I"/"me"/"our" or for the user's own words like "docs"/"report"/"notes" \
("docs" must NOT become "documentation").
- Never invent, substitute, or answer a different question. If you are unsure \
how to fix something, leave it unchanged rather than guessing a replacement.
- Never invent a specific technology/database type (SQL, MongoDB, PostgreSQL, \
etc.) that the user did not mention. "our database" must stay "our database", \
not become "your SQL database".
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

Example:
Input: "What is the total number of records in our database?"
Output: "What is the total number of records in our database?"

Example (do NOT do this — this invents a different question):
Input: "Who won the most recent Formula 1 race?"
WRONG Output: "Who is the most successful driver among the top five drivers of all time?"
RIGHT Output: "Who won the most recent Formula 1 race?"
"""

# 2026-07-25: replaced the old fixed MAX_HISTORY_TURNS=3 cap with a
# token-budgeted selection. ~800 tokens is a conservative slice of
# fast_llm's 2048-token default context window: REWRITE_SYSTEM_PROMPT
# itself runs roughly 400 tokens, plus the current question and
# formatting overhead, leaving this as the safe remaining share for
# prior-turn context specifically.
HISTORY_PROMPT_TOKEN_BUDGET = 800

# 2026-08-11 fix (Test 4): added "our" -- see module docstring. Previously
# only "my"/"i"/"me" were covered, so a rewrite that dropped/altered "our"
# (e.g. "our database" -> "your SQL database") was never caught here.
_PERSONAL_REF = re.compile(r"\b(my|our|i|me)\b", re.IGNORECASE)
_QUANTIFIER_RE = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b", re.IGNORECASE)
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")
_WORD_RE = re.compile(r"[a-z0-9]+")

# 2026-08-11 fix (Test 4): new backstop, same pattern as
# _dropped_or_altered_acronym below. Catches a rewrite inventing a
# specific database/tech TYPE the user never named -- the root cause of
# "our database" -> "your SQL database". Deliberately a fixed list of
# common DB/tech type names rather than a broader heuristic, matching
# this file's established style of narrow, high-precision backstops.
_INVENTED_TECH_TYPE_RE = re.compile(
    r"\b(sql|postgresql|postgres|mysql|mongodb|nosql|redis|graphql|"
    r"sqlite|oracle|mariadb|cassandra|dynamodb|firebase|supabase)\b",
    re.IGNORECASE,
)

# Minimal function-word list for word-overlap comparisons. Deliberately
# short — this only exists to stop overlap ratios being inflated by
# words like "the"/"is"/"what" that nearly every question shares
# regardless of topic. Content words (nouns, numerals, domain terms)
# are what should drive the comparison.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "whom", "how", "why", "when", "where",
    "does", "do", "did", "this", "that", "these", "those",
    "of", "in", "on", "to", "for", "and", "or", "as", "with", "from",
    "by", "at", "it", "its", "according", "paper", "please",
}

# Below this overall similarity ratio, a rewrite is treated as having
# replaced the question's actual content rather than just cleaning it
# up. Tune based on observed false positives/negatives in logs.
_DIVERGENCE_SIMILARITY_THRESHOLD = 0.4

# A rewrite is flagged as "resembles a prior turn instead of the
# current question" if its word-overlap with that prior turn exceeds
# its word-overlap with the current original question by at least this
# margin, AND the prior-turn overlap itself clears the minimum below.
# The margin (not an absolute cutoff alone) is what makes this
# paraphrase-tolerant: legitimate follow-up rewrites can share real
# overlap with recent history (that's the point of context resolution)
# as long as they still resemble the current question at least as
# much. Needs tuning against continued eval observations.
_PRIOR_TURN_OVERLAP_MARGIN = 0.10
_PRIOR_TURN_MIN_OVERLAP = 0.35


def _select_recent_history(history: list[dict], token_budget: int) -> list[dict]:
    """
    Selects the most recent turns that fit within token_budget,
    scanning from the newest turn backwards and stopping before the
    running estimated token count would exceed the budget.

    Always keeps at least the single most recent turn, even if that
    one turn alone exceeds budget — dropping the current context
    entirely would remove the very thing this agent needs to resolve
    "what about X?"-style follow-ups against.
    """
    selected: list[dict] = []
    running = 0
    for turn in reversed(history):
        t = estimate_tokens(turn.get("content", ""))
        if selected and running + t > token_budget:
            break
        selected.append(turn)
        running += t
    return list(reversed(selected))


def _format_history(history: list[dict], summaries: list[dict]) -> str:
    """
    Builds the history block shown to the model: long-term summaries
    (older content, compressed) first, then as many recent verbatim
    turns as fit inside HISTORY_PROMPT_TOKEN_BUDGET.
    """
    if not history and not summaries:
        return "(no prior conversation)"

    parts = []

    if summaries:
        summary_lines = "\n".join(
            f"- {s.get('summary', '')}" for s in summaries if s.get("summary")
        )
        if summary_lines:
            parts.append(f"Earlier conversation summary:\n{summary_lines}")

    recent = _select_recent_history(history, HISTORY_PROMPT_TOKEN_BUDGET)
    if recent:
        lines = [f"{t.get('role', 'user')}: {t.get('content', '')}" for t in recent]
        parts.append("Recent conversation:\n" + "\n".join(lines))

    return "\n\n".join(parts) if parts else "(no prior conversation)"


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

    If the user's original question contains "my"/"our"/"I"/"me" and the
    rewritten version doesn't contain any of them, the rewrite has
    stripped (or silently swapped, e.g. "our" -> "your") a signal the
    Planner depends on for routing to ["documents"]/["database"], and
    which downstream generation depends on to avoid inventing an
    unstated ownership/type. Deliberately narrow: only this one known
    failure mode, to avoid false-positiving on legitimate rewrites.

    2026-08-11: added "our" to the underlying _PERSONAL_REF regex (was
    previously "my"/"i"/"me" only) -- see module docstring, Test 4.
    """
    return bool(_PERSONAL_REF.search(original)) and not _PERSONAL_REF.search(rewritten)


def _dropped_quantifier(original: str, rewritten: str) -> bool:
    """
    Catches silent numeral/quantifier loss — e.g. "What two techniques
    are combined...?" rewritten as "Summarize the technique used...",
    dropping "two" (and pluralizing away the fact that multiple
    techniques were asked about). This is subtler than outright
    hallucination: the rewrite is topically on-target and won't trip
    _diverged_too_much, but it silently changes the actual claim being
    asked about, which can bias retrieval toward a single technique
    instead of both.

    Every quantifier word/digit present in the original must survive
    into the rewrite unchanged (order doesn't matter).
    """
    original_q = set(m.lower() for m in _QUANTIFIER_RE.findall(original))
    if not original_q:
        return False
    rewritten_q = set(m.lower() for m in _QUANTIFIER_RE.findall(rewritten))
    return not original_q.issubset(rewritten_q)


def _dropped_or_altered_acronym(original: str, rewritten: str) -> bool:
    """
    Code-level backstop for the "never expand acronyms" prompt rule.

    Observed failure: "What does FNOL stand for?" rewritten to "What is
    the abbreviation for 'federal net of losses'?" — the acronym itself
    disappears, replaced by a hallucinated (and wrong) expansion. The
    prompt already forbids this, but this is a fast model and, per this
    project's established pattern (see _dropped_personal_reference),
    prompt-only guarantees aren't reliable enough for something this
    consequential to retrieval.

    Every all-caps 2-6 letter token in the original must appear
    verbatim, unchanged, in the rewrite.
    """
    original_acronyms = set(_ACRONYM_RE.findall(original))
    if not original_acronyms:
        return False
    rewritten_acronyms = set(_ACRONYM_RE.findall(rewritten))
    return not original_acronyms.issubset(rewritten_acronyms)


def _invented_tech_type(original: str, rewritten: str) -> bool:
    """
    2026-08-11 fix (Test 4 root cause): code-level backstop for a
    rewrite that invents a specific database/technology TYPE the user
    never named.

    Observed failure: "What is the total number of records in our
    database?" rewritten to "How many records are there in your SQL
    database?" -- "SQL" appears nowhere in the original question. The
    prompt already forbids this (see REWRITE_SYSTEM_PROMPT's explicit
    rule + example), but per this file's established pattern (see
    _dropped_personal_reference, _dropped_or_altered_acronym),
    prompt-only compliance isn't trusted alone for anything this
    consequential -- an invented tech type can bias the Planner/
    downstream tooling toward a specific (wrong, unstated) database
    technology.

    Any tech-type word appearing in the rewrite but NOT in the original
    is treated as invented, regardless of which specific word it is.
    """
    orig_types = set(m.lower() for m in _INVENTED_TECH_TYPE_RE.findall(original))
    new_types = set(m.lower() for m in _INVENTED_TECH_TYPE_RE.findall(rewritten))
    return bool(new_types - orig_types)


def _diverged_too_much(original: str, rewritten: str) -> bool:
    """
    Broader safety net, complementary to _dropped_personal_reference.

    Catches cases where the model replaces the question's actual
    content with a different (but topically related) question. Uses
    overall string similarity rather than matching specific
    words/phrases, since the failure isn't limited to one vocabulary
    pattern.

    Runs unconditionally regardless of whether history is present —
    see module docstring's 2026-07-04 bug note for why the old
    `not history` gate let a hallucinated rewrite through during a
    multi-turn eval session.
    """
    from difflib import SequenceMatcher
    ratio = SequenceMatcher(None, original.lower(), rewritten.lower()).ratio()
    return ratio < _DIVERGENCE_SIMILARITY_THRESHOLD


def _word_set(text: str) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _overlap_ratio(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _matches_prior_turn(original: str, rewritten: str, history: list[dict]) -> tuple[bool, str]:
    """
    True if the rewrite resembles a PRIOR turn (user OR assistant) in
    history more than it resembles the CURRENT question it's supposed
    to be a rewrite of — e.g. the model regurgitated an earlier
    question, or the previous answer, instead of engaging with the
    current turn.

    Word-overlap based (order/reordering/preamble-tolerant) rather than
    character-sequence based, and relative rather than an absolute
    threshold: a legitimate context-resolution rewrite can legitimately
    share real vocabulary with recent history (that's the point of
    resolving "what about X?" using a prior turn), so what actually
    distinguishes a hallucination is that it resembles some PAST turn
    MORE than it resembles the CURRENT question — a correct rewrite is
    derived from the current question and should never lose that
    comparison.

    Checks against the FULL history list passed in — not just the
    token-trimmed subset used for the prompt itself (see
    _select_recent_history) — since this safety check should compare
    against everything actually stored for this session, regardless of
    what fit in this particular prompt.

    Returns (matched, matched_turn_content) for logging.
    """
    original_words = _word_set(original)
    rewritten_words = _word_set(rewritten)
    own_overlap = _overlap_ratio(original_words, rewritten_words)

    for turn in history:
        role = turn.get("role")
        if role not in ("user", "assistant"):
            continue
        content = turn.get("content", "")
        if not content:
            continue
        prior_words = _word_set(content)
        prior_overlap = _overlap_ratio(prior_words, rewritten_words)

        if (
            prior_overlap >= _PRIOR_TURN_MIN_OVERLAP
            and prior_overlap > own_overlap + _PRIOR_TURN_OVERLAP_MARGIN
        ):
            return True, content

    return False, ""


class RewriterAgent(BaseAgent):
    """
    Resolves conversational context and normalizes spelling/grammar
    before planning/retrieval. See module docstring for design notes.
    """

    async def _execute(self, state: AgentState) -> AgentState:
        original_question = state.question

        # ---- Fetch history + long-term summaries ----
        # 2026-07-25: switched from short_term.get_history() (raw,
        # summary-blind) to load_context() so long-term summaries of
        # older, already-evicted turns are actually included instead of
        # silently missing from every rewrite.
        history = []
        summaries = []
        if mm_module.memory_manager:
            try:
                context = await mm_module.memory_manager.load_context(
                    state.user_id, state.session_id
                )
                history = context.get("history", [])
                summaries = context.get("summaries", [])
            except Exception as e:
                print(f"[REWRITER] Failed to fetch context, proceeding without it: {e}")
                history, summaries = [], []
        else:
            print("[REWRITER] memory_manager not initialized, proceeding without history")

        history_text = _format_history(history, summaries)

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
                f'[REWRITER] Rewrite dropped personal reference ("my"/"our"/"I"/"me"), '
                f'falling back to original question. Bad rewrite was: "{rewritten}"'
            )
            rewritten = ""

        if rewritten and _dropped_quantifier(original_question, rewritten):
            print(
                f'[REWRITER] Rewrite dropped a numeral/quantifier present in the '
                f'original, falling back to original question. Bad rewrite was: "{rewritten}"'
            )
            rewritten = ""

        if rewritten and _dropped_or_altered_acronym(original_question, rewritten):
            print(
                f'[REWRITER] Rewrite dropped or altered an acronym present in the '
                f'original, falling back to original question. Bad rewrite was: "{rewritten}"'
            )
            rewritten = ""

        if rewritten and _invented_tech_type(original_question, rewritten):
            print(
                f'[REWRITER] Rewrite invented a technology/database type not '
                f'present in the original, falling back to original question. '
                f'Bad rewrite was: "{rewritten}"'
            )
            rewritten = ""

        if rewritten and _diverged_too_much(original_question, rewritten):
            print(
                f'[REWRITER] Rewrite diverged too far from original (similarity below '
                f'{_DIVERGENCE_SIMILARITY_THRESHOLD}), falling back to original question. '
                f'Bad rewrite was: "{rewritten}"'
            )
            rewritten = ""

        if rewritten and history:
            matched, matched_content = _matches_prior_turn(original_question, rewritten, history)
            if matched:
                print(
                    f'[REWRITER] Rewrite resembles a PRIOR turn more than the current '
                    f'question, falling back to original question. Bad rewrite was: '
                    f'"{rewritten}" | Matched prior turn: "{matched_content[:100]}"'
                )
                rewritten = ""

        if rewritten and rewritten != original_question:
            print(f'[REWRITER] "{original_question}" -> "{rewritten}"')

        state.rewritten_question = rewritten
        return state