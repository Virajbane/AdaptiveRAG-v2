from app.agents.base import BaseAgent
from app.agents.state import AgentState
from app.agents.prompts import ANSWER_PROMPT, DIRECT_LLM_PROMPT
from app.utils.tokenization import count_tokens
import re

# 2026-08-01: real context-window budget check.
#
# Reserved for the model's own generation. This bounds how much of
# num_ctx we treat as "spoken for" by the answer itself, so we never
# compute an "available for context" number by pretending the LLM will
# generate 0 tokens back. Also passed explicitly to call_llm() below so
# what we budget against matches what's actually requested.
ANSWER_RESERVED_OUTPUT_TOKENS = 500

# Slack for tokenizer-estimate error (mainly relevant on the Groq path,
# where counting is a chars-per-token estimate, not an exact count).
# Kept out of the "usable" budget so a slight undercount doesn't tip
# us over num_ctx in practice.
BUDGET_SAFETY_MARGIN_TOKENS = 50

# Excludes digits embedded inside hyphenated/alphanumeric identifiers
# (e.g. "StepAudio-2-mini", "GPT-5-Duplex") -- those aren't factual
# claims to verify, they're part of a proper noun. Confirmed root cause
# of a false-decline: "StepAudio-2-mini" (a correct answer) was rejected
# because the embedded "2" was treated as an unverified numeric claim.
_NUM_RE = re.compile(r'(?<![A-Za-z0-9-])\d+(?:\.\d+)?%?(?![A-Za-z0-9-])')

# 2026-08-04 fix (Q16 root cause): matches a single "label=value" table
# field, e.g. "FullDuplexBench 1.5.Lat. ↓=826" as produced by
# DoclingChunker._chunk_table(). Table rows are comma-separated on a
# single line ("Row [Lychee-FD]: Stop.=570, Lat.=826, ..."), so without
# this, the row-level span check below treats every field in the row as
# one span -- see _numeric_claims_grounded's docstring for why that's
# not fine-grained enough.
_KV_RE = re.compile(r'([^,]+?=[^,]+)')

# Matches a DoclingChunker table-row line, e.g.
# "Row [Lychee-FD (Ours)]: FDBench.SRR ↑=86.3, ..." -- captures the
# entity label and the rest of the fields separately. Table rows are
# emitted as ONE line (no embedded newlines), which matters below: this
# line must NOT be run through the prose sentence-splitter, because
# DoclingChunker's own column names contain periods that aren't sentence
# boundaries (e.g. "FullDuplexBench 1.5.Lat."). Splitting a table row on
# '.' the same way prose is split shreds it into meaningless fragments.
_ROW_LINE_RE = re.compile(r'^\s*Row \[(.*?)\]:\s*(.*)$')

_QUESTION_STOPWORDS = {
    "how", "what", "which", "who", "whom", "when", "where", "why",
    "is", "are", "was", "were", "does", "did", "do", "the", "a", "an",
}

# 2026-08-10 FIX (Test 9 — Field-level grounding): Matches metric/field names
# asked about in the question. Ensures numeric claims are attributed not just
# to the correct entity, but to the correct field (e.g., "S→S" vs "S→T").
_FIELD_HINT_RE = re.compile(
    r"\b(s\s*[-→>]+\s*s|s\s*[-→>]+\s*t|utmos|bleu|rouge|f1|accuracy|precision|"
    r"recall|latency|lat\.|stop\.|wer|srr|sir)\b", re.IGNORECASE
)


def _extract_field_hint(question: str) -> str | None:
    """
    Extract the metric/field name from the question (e.g., "S→S", "UTMOS").
    Used in _numeric_claims_grounded to enforce that numeric claims are
    attributed to the correct field, not just the correct entity.

    Returns:
        The matched field hint (normalized to remove spaces), or None
    """
    m = _FIELD_HINT_RE.search(question)
    return m.group(0).replace(" ", "") if m else None


# 2026-08-09 FIX: Each tool result type gets a formatter so AnswerAgent
# can include calculator/weather/slack/email results, not just web.
# Returns None for missing/errored results so they're excluded.
#
# 2026-08-22 FIX: added a "database" case. Previously this function had
# no branch for "database" at all, matching the missing "database" check
# in AnswerAgent._execute below -- a successful database tool result
# was silently never turned into evidence text, so the answer agent
# always fell through to "no usable content" even when tool_agent had
# already found the right answer.
def _format_tool_result(kind: str, result: dict | None) -> str | None:
    if not result or result.get("error"):
        return None

    if kind == "calculator":
        value = result.get("result")
        if value is None:
            return None
        expression = result.get("expression")
        prefix = f"{expression} = " if expression else ""
        return f"[Calculator Result] {prefix}{value}"

    if kind == "weather":
        temp = result.get("temperature")
        desc = result.get("description")
        if temp is None and not desc:
            return None
        location = result.get("location") or "the requested location"
        details = ", ".join(
            p for p in (f"{temp}°C" if temp is not None else None, desc) if p
        )
        return f"[Weather Result] {location}: {details}"

    if kind == "slack":
        channel = result.get("channel") or "the requested channel"
        return f"[Slack Result] Message was posted to {channel}."

    if kind == "email":
        to_email = result.get("to_email") or "the requested recipient"
        return f"[Email Result] Email was sent to {to_email}."

    if kind == "database":
        entity = result.get("entity") or "records"
        query_type = result.get("query_type")
        count = result.get("count")
        rows = result.get("rows") or []
        if query_type == "count":
            if count is None:
                return None
            return f"[Database Result] {entity}: {count} record(s)."
        if query_type == "latest":
            if not rows:
                return None
            return f"[Database Result] Latest {entity}: {rows[0]}"
        if query_type == "list":
            if not rows:
                return None
            return f"[Database Result] {entity} ({count} total): {rows}"
        return None

    return None


# =============================================================================
# 2026-08-06 FIX #2: IMPROVED ENTITY EXTRACTION & GROUNDING
# =============================================================================
# Previously: extracted entities from question (capitalization heuristic)
#             This missed aliases like "Lychee-FD (Ours)" and failed on pronouns
#
# Now: extract entities from ACTUAL ROW LABELS in the context
#      Fallback to question-based extraction only if no table rows present
# =============================================================================

def _extract_entities_from_rows(context: str) -> set:
    """
    Extract entity labels directly from table row lines in the context.

    This is more reliable than extracting from the question because:
    - Captures entity aliases exactly as they appear in data ("Lychee-FD (Ours)")
    - Handles questions with pronouns ("What does it achieve?" has no entities)
    - Won't be fooled by entity names in other parts of the prose

    Returns a set of entity labels like {"Lychee-FD (Ours)", "Moshi", "Baseline"}
    """
    entities = set()
    for line in context.split("\n"):
        row_match = _ROW_LINE_RE.match(line)
        if row_match:
            row_entity = row_match.group(1).strip()
            if row_entity:
                entities.add(row_entity)
    return entities


def _extract_entities_from_question(question: str) -> set:
    """
    Fallback: extract capitalized multi-char tokens from the question.
    Used only if no table rows are present in the context.

    This is the original approach, kept for non-table questions.
    """
    return {
        e for e in re.findall(r"\b[A-Z][A-Za-z0-9\-]{2,}\b", question)
        if e.lower() not in _QUESTION_STOPWORDS
    }


def _numeric_claims_grounded(answer: str, context: str, question: str, sources: list[str] = None) -> bool:
    """
    Verify that numbers in the answer are grounded in the context.
    
    STAGE 14 FIX: SOURCE-AWARE grounding.
    
    Different sources have different validation requirements:
    - calculator / weather / tool / database: DETERMINISTIC (no grounding required)
    - direct_llm: GENERAL KNOWLEDGE (no grounding required)
    - documents / web: RETRIEVAL-BASED (full grounding required)

    2026-08-06 fix (entity-attribution cascade):
    - Extract entities from ACTUAL ROW LABELS in context, not from question
    - For table rows, enforce that number + entity + field all cooccur in
      the same row's own field-level span
    - For prose, use the weaker sentence-level span check (no per-field split)

    2026-08-10 fix (field-level grounding for Test 9):
    - Also enforce that the field name asked about (e.g., "S→S", "UTMOS")
      appears in the span containing the number and entity.
    - Catches cases like "84.1 for Lychee-FD but wrong field (S→T not S→S)"

    This catches the "4.50 for Lychee-FD but you answered it for Moshi" bug
    by requiring an answer's cited number to appear in a span that ALSO
    contains the queried entity's row label.

    Args:
        answer: The generated answer text
        context: The context provided to the LLM
        question: The original user question
        sources: List of sources used (from state.sources_needed)

    Returns:
        True if all numbers in the answer are properly grounded and attributed
        False if any number is missing from context or attributed to wrong entity/field
    """
    sources = sources or []
    
    # ── STAGE 14 FIX: Deterministic sources don't require grounding ─────
    # If the source is deterministic/authoritative, trust the answer as-is.
    # Examples:
    #   - calculator: 355 is correct by construction (480 - 125 = 355)
    #   - weather: 22.28°C is correct from the weather API
    #   - database: results are correct from the database query
    #   - direct_llm: general knowledge doesn't need document evidence
    if sources and sources[0] in ("calculator", "weather", "tool", "database", "direct_llm"):
        return True
    numbers_in_answer = _NUM_RE.findall(answer)
    if not numbers_in_answer:
        return True

    # Extract entities from ROWS first (more precise than question parsing)
    # This is the key change: use actual row labels instead of capitalization heuristic
    entities_from_rows = _extract_entities_from_rows(context)

    if not entities_from_rows:
        # No table rows in context -- fall back to question-based extraction
        # for conceptual/non-comparative questions (unchanged behavior)
        entities = _extract_entities_from_question(question)
        if not entities:
            # No named entity anywhere -- weaker existence check
            # For purely conceptual questions with no entity names
            return all(num in context for num in numbers_in_answer)
    else:
        # Use entities extracted from actual row labels
        entities = entities_from_rows

    # Build fine-grained spans: table rows split on commas (per field),
    # prose split on sentence boundaries
    fine_spans = []
    for line in context.split("\n"):
        row_match = _ROW_LINE_RE.match(line)
        if row_match:
            row_entity = row_match.group(1)
            fields_text = row_match.group(2)
            kv_fragments = _KV_RE.findall(fields_text)
            if kv_fragments:
                # Each field gets its own span with the row label attached
                fine_spans.extend(
                    f"Row [{row_entity}]: {frag}" for frag in kv_fragments
                )
                continue

        # Non-table line: split on sentence boundaries (skip decimals)
        for sentence in re.split(r"(?<!\d)\.(?!\d)", line):
            if sentence.strip():
                fine_spans.append(sentence)

    # Validate each number in the answer
    for num in numbers_in_answer:
        # For each number, find ALL spans containing it
        spans_with_num = [s for s in fine_spans if num in s]

        if not spans_with_num:
            # Number doesn't appear anywhere in context
            return False

        # Check if any of those spans also contain one of the queried entities
        num_is_attributed = any(
            any(e.lower() in s.lower() for e in entities)
            for s in spans_with_num
        )

        # 2026-08-10 FIX: Also check field name if present in question
        # This prevents accepting a number that's attributed to the right entity
        # but the wrong field (e.g., 84.1 for Lychee-FD S→T when asked for S→S)
        field_hint = _extract_field_hint(question)
        if field_hint and num_is_attributed:
            num_is_attributed = any(
                field_hint.lower() in s.lower().replace(" ", "")
                for s in spans_with_num if any(e.lower() in s.lower() for e in entities)
            )

        if not num_is_attributed:
            # Number exists in context but not attached to any queried entity
            # or (after 2026-08-10 fix) not attached to the correct field
            return False

    return True


class AnswerAgent(BaseAgent):
    async def _execute(self, state: AgentState) -> AgentState:

        # 2026-08-09 FIX: Respect routing decision. Only include evidence
        # from sources that were actually routed to by Planner.
        sources_needed = state.sources_needed or []

        # 2026-08-10 FIX (Test 5 — direct_llm handler): Handle direct_llm BEFORE
        # the evidence-sufficiency gate. direct_llm questions have NO documents or
        # tools by design — they must be answered from the LLM's own knowledge.
        # If we don't handle this first, the gate below unconditionally declines
        # because (correctly) there's no doc context, and the critic then hallucinates.
        if sources_needed == ["direct_llm"]:
            try:
                prompt = DIRECT_LLM_PROMPT.format(question=state.question)
                response = await self.call_llm(prompt, max_tokens=ANSWER_RESERVED_OUTPUT_TOKENS)
                state.answer = response.strip()
                state.sources = []
                # Use planner's routing confidence; critic evaluates plausibility, not grounding
                state.confidence_final = state.confidence
                print(f"[ANSWER] direct_llm: generated answer without document/tool evidence")
                return state
            except Exception as e:
                print(f"[ANSWER] direct_llm FAILED: {type(e).__name__}: {e}")
                state.error = f"Answer agent error: {str(e)}"
                state.answer = "Sorry, I couldn't generate an answer."
                state.sources = []
                state.confidence_final = 0.0
                return state

        # ── Document evidence ────────────────────────────────────────
        # Include top_docs only if "documents" was routed to
        docs_wanted = "documents" in sources_needed
        if docs_wanted:
            protected = [d for d in state.retrieved_docs if d.get("protected")]
            other = sorted(
                (d for d in state.retrieved_docs if not d.get("protected")),
                key=lambda d: d.get("rerank_score", d.get("combined_score", 0.0)),
                reverse=True,
            )
            top_docs = (protected + other)[:3]
        else:
            top_docs = []

        # ── Tool results ─────────────────────────────────────────────
        # Include tool results only if corresponding source was routed to.
        # "tool" source includes weather, email, slack; "calculator" and
        # "database" are their own sources.
        tool_context_parts = []
        web_result_count = 0
        non_web_tool_used = False

        # Web search: only if "web" was routed
        web_result = None
        if "web" in sources_needed:
            web_result = state.tool_results.get("web_search") if state.tool_results else None
            if web_result and "error" not in web_result:
                entries = web_result.get("results", [])
                web_result_count = len(entries)
                if entries:
                    formatted = []
                    for i, entry in enumerate(entries, 1):
                        title = entry.get("title", "")
                        snippet = entry.get("snippet") or entry.get("content") or ""
                        url = entry.get("url", "")
                        formatted.append(f"[Web {i}] {title}\n{snippet}\n{url}".strip())
                    tool_context_parts.append("\n\n".join(formatted))

        # Calculator: only if "calculator" was routed
        if "calculator" in sources_needed:
            result = state.tool_results.get("calculator") if state.tool_results else None
            formatted = _format_tool_result("calculator", result)
            if formatted:
                tool_context_parts.append(formatted)
                non_web_tool_used = True

        # Tool-executed results (weather, email, slack): only if "tool" was routed
        if "tool" in sources_needed:
            for kind in ("weather", "slack", "email"):
                result = state.tool_results.get(kind) if state.tool_results else None
                formatted = _format_tool_result(kind, result)
                if formatted:
                    tool_context_parts.append(formatted)
                    non_web_tool_used = True

        # Database: only if "database" was routed.
        #
        # 2026-08-22 FIX: this branch didn't exist before. A successful
        # database tool result (state.tool_results["database"]) was
        # produced by tool_agent.py but never read here, so tool_context
        # stayed empty for every database question and the evidence-
        # sufficiency gate below always declined with "no usable content"
        # regardless of whether the query actually succeeded.
        if "database" in sources_needed:
            result = state.tool_results.get("database") if state.tool_results else None
            formatted = _format_tool_result("database", result)
            if formatted:
                tool_context_parts.append(formatted)
                non_web_tool_used = True

        # 2026-08-09 FIX: Metadata as evidence. Metadata is treated like
        # retrieved evidence, not as a shortcut bypass. If Planner identified
        # metadata (e.g., "author", "title"), include it in the evidence block
        # so the LLM can reason over it.
        if state.metadata_answer:
            meta_text = "\n".join(
                f"{k}: {v}" for k, v in state.metadata_answer.items()
            )
            if meta_text:
                tool_context_parts.insert(0, f"[Metadata]\n{meta_text}")

        tool_context = "\n\n".join(tool_context_parts)

        # ── Evidence sufficiency check ───────────────────────────────
        if not top_docs and not tool_context:
            web_error = web_result.get("error") if web_result else None
            has_any_retrieved_docs = bool(state.retrieved_docs)

            if not has_any_retrieved_docs and not web_error:
                message = (
                    "I don't have any documents to search. "
                    "Please upload documents first, then ask your question."
                )
            elif has_any_retrieved_docs and web_error:
                message = (
                    "I found some content in your documents, but it wasn't a strong "
                    "enough match to answer confidently, and the web search I tried "
                    "as backup failed due to a connection error. Try rephrasing, or "
                    "ask again in a moment."
                )
            elif has_any_retrieved_docs and not web_error:
                message = (
                    "I found some content in your documents, but it wasn't a strong "
                    "enough match to confidently answer this question."
                )
            else:  # no retrieved docs, web errored
                message = (
                    "I don't have relevant documents for this, and the web search I "
                    "tried failed due to a connection error. Please try again."
                )

            state.answer = message
            state.sources = []
            state.confidence_final = state.confidence * 0.3
            print(f"[ANSWER] No usable content — sources_needed={sources_needed}, "
                  f"has_docs={has_any_retrieved_docs}, web_error={bool(web_error)}")
            return state

        # ── Context-window budget check ──────────────────────────────
        use_groq = getattr(self.llm, "use_groq", False)
        num_ctx = getattr(self.llm, "num_ctx", 2048)

        scaffold_tokens = count_tokens(
            ANSWER_PROMPT.format(question=state.question, context=""),
            use_groq=use_groq,
        )
        available_for_context = (
            num_ctx - ANSWER_RESERVED_OUTPUT_TOKENS - scaffold_tokens
            - BUDGET_SAFETY_MARGIN_TOKENS
        )

        if available_for_context <= 0:
            print(
                f"[ANSWER][BUDGET] num_ctx={num_ctx} leaves NO room for "
                f"context (scaffold={scaffold_tokens}, reserved_output="
                f"{ANSWER_RESERVED_OUTPUT_TOKENS}, margin="
                f"{BUDGET_SAFETY_MARGIN_TOKENS}). Proceeding with empty "
                f"context -- consider raising num_ctx or lowering "
                f"ANSWER_RESERVED_OUTPUT_TOKENS."
            )
            available_for_context = 0

        # Priority-ordered candidate blocks
        blocks = []
        for i, doc in enumerate(top_docs, 1):
            text = f"[Source {i}]\n{doc['text']}"
            blocks.append({
                "text": text,
                "tokens": count_tokens(text, use_groq=use_groq),
                "protected": bool(doc.get("protected")),
                "kind": "doc",
                "doc": doc,
            })
        if tool_context:
            for entry in tool_context.split("\n\n"):
                blocks.append({
                    "text": entry,
                    "tokens": count_tokens(entry, use_groq=use_groq),
                    "protected": False,
                    "kind": "tool",
                    "doc": None,
                })

        included_blocks = []
        dropped_blocks = []
        used_tokens = 0
        for block in blocks:
            fits = used_tokens + block["tokens"] <= available_for_context
            if fits or block["protected"]:
                included_blocks.append(block)
                used_tokens += block["tokens"]
                if not fits:
                    print(
                        f"[ANSWER][BUDGET] protected doc included despite "
                        f"exceeding budget ({block['tokens']} tokens, "
                        f"available was {available_for_context - (used_tokens - block['tokens'])})"
                    )
            else:
                dropped_blocks.append(block)

        if dropped_blocks:
            print(
                f"[ANSWER][BUDGET] dropped {len(dropped_blocks)} block(s) "
                f"to fit num_ctx={num_ctx}: "
                + ", ".join(f"{b['kind']}({b['tokens']} tok)" for b in dropped_blocks)
            )

        # Keep top_docs/tool_context in sync with what's actually sent
        top_docs = [b["doc"] for b in included_blocks if b["kind"] == "doc"]
        doc_context = "\n\n".join(b["text"] for b in included_blocks if b["kind"] == "doc")
        tool_context = "\n\n".join(b["text"] for b in included_blocks if b["kind"] == "tool")

        context_parts = [p for p in (doc_context, tool_context) if p]
        context = "\n\n".join(context_parts) if context_parts else "(no context available)"

        # Preserve the evidence boundary after every selection/truncation
        # decision. `retrieved_docs` remains the broader retrieval result;
        # these fields represent exactly what the answer LLM can use.
        # They are internal diagnostics, intentionally kept out of the
        # public orchestrator response.
        state.answer_context = context
        state.answer_context_docs = [dict(doc) for doc in top_docs]
        state.answer_context_dropped_docs = [
            dict(block["doc"])
            for block in dropped_blocks
            if block["kind"] == "doc" and block["doc"] is not None
        ]

        evidence_desc = []
        if top_docs:
            evidence_desc.append(f"{len(top_docs)} document(s)")
        if web_result_count:
            evidence_desc.append(f"{web_result_count} web result(s)")
        if non_web_tool_used:
            evidence_desc.append("tool result(s)")
        
        print(
            f"[ANSWER] Using {', '.join(evidence_desc) if evidence_desc else 'no evidence'} "
            f"| sources_needed={sources_needed} "
            f"| budget: scaffold={scaffold_tokens} used={used_tokens}/"
            f"{available_for_context} num_ctx={num_ctx}"
        )

        try:
            prompt = ANSWER_PROMPT.format(
                question=state.question,
                context=context
            )

            response = await self.call_llm(prompt, max_tokens=ANSWER_RESERVED_OUTPUT_TOKENS)
            state.answer = response.strip()

            # 2026-08-06 fix: Grounding verification with row/entity protection
            # STAGE 14 FIX: Pass sources so grounding is source-aware
            declined_on_ungrounded_number = not _numeric_claims_grounded(
                state.answer, context, state.question, state.sources_needed
            )
            if declined_on_ungrounded_number:
                print(
                    f"[ANSWER] Numeric claim not found verbatim in context "
                    f"(or not attributed to the entity/field asked about) "
                    f"— declining rather than shipping unverified number. "
                    f"Original answer was: {state.answer!r}"
                )
                state.answer = (
                    "I found related content in the document, but couldn't verify "
                    "a specific number for this with confidence from the retrieved "
                    "text. This may be a figure or chart value that wasn't "
                    "extracted as readable text, or the number belongs to a "
                    "different entity or field than the one asked about."
                )

            # Source cards: only include documents that were actually sent to LLM
            state.sources = [
                {
                    "doc_id": doc["doc_id"],
                    "chunk_index": doc["chunk_index"],
                    "filename": doc.get("filename") or f"Source {i}",
                    "text": doc["text"][:200],
                    "score": doc.get("rerank_score", doc.get("combined_score", 0.0)),
                }
                for i, doc in enumerate(top_docs, 1)
            ]

            # Add web results to sources if they were used
            if web_result and "error" not in web_result:
                state.sources.extend([
                    {
                        "doc_id": None,
                        "chunk_index": None,
                        "filename": entry.get("title") or entry.get("url") or f"Web result {i}",
                        "text": (entry.get("snippet") or entry.get("content") or "")[:200],
                        "score": None,
                        "url": entry.get("url"),
                    }
                    for i, entry in enumerate(web_result.get("results", []), 1)
                ])

            # ── Confidence scoring ───────────────────────────────────
            # Use top_docs (what was actually sent), not all retrieved_docs
            if top_docs:
                scored_docs = [
                    doc.get("rerank_score", doc.get("combined_score", 0.0))
                    for doc in top_docs
                ]
                avg_doc_score = sum(scored_docs) / len(scored_docs) if scored_docs else 0.0
                avg_doc_score = max(0.0, min(avg_doc_score, 1.0))
                state.confidence_final = min(
                    state.confidence * 0.5 + avg_doc_score * 0.5, 1.0
                )
            elif non_web_tool_used:
                # Tool-only answer is deterministic/exact
                state.confidence_final = max(state.confidence, 0.85)
            else:
                # Web-only or LLM-only
                state.confidence_final = state.confidence

            if declined_on_ungrounded_number:
                state.confidence_final = min(state.confidence_final, 0.3)

            print(f"[ANSWER] Generated response with {len(state.sources)} source(s)")
            print(f"[ANSWER] Confidence: {state.confidence_final:.2f}")

        except Exception as e:
            print(f"[ANSWER] Answer generation FAILED: {type(e).__name__}: {e}")
            state.error = f"Answer agent error: {str(e)}"
            state.answer = "Sorry, I couldn't generate an answer."
            state.sources = []
            state.confidence_final = 0.0

        return state