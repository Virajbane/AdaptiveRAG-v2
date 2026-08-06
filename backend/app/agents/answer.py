from app.agents.base import BaseAgent
from app.agents.state import AgentState
from app.agents.prompts import ANSWER_PROMPT
from app.utils.tokenization import count_tokens
import re

# 2026-08-01: real context-window budget check.
#
# Reserved for the model's own generation. This bounds how much of
# num_ctx we treat as "spoken for" by the answer itself, so we never
# compute an "available for context" number by pretending the LLM will
# generate 0 tokens back. Also passed explicitly to call_llm() below so
# what we budget against matches what's actually requested -- there
# was previously a DEFAULT_NUM_CTX=2048 / default generate() max_tokens
# =2000 mismatch (see provider.py) that left as little as ~48 tokens of
# real headroom for the entire prompt, regardless of what this file did
# with chunk sizes. This value is generous for this prompt's own rules
# (mostly one-sentence or short-list answers; see ANSWER_PROMPT), while
# leaving real room for retrieved context.
ANSWER_RESERVED_OUTPUT_TOKENS = 500

# Slack for tokenizer-estimate error (mainly relevant on the Groq path,
# where counting is a chars-per-token estimate, not an exact count --
# see app/utils/tokenization.py). Kept out of the "usable" budget so a
# slight undercount doesn't tip us over num_ctx in practice.
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
# '.' the same way prose is split shreds it into meaningless fragments
# *before* the comma/field split ever runs -- confirmed by direct testing
# against real Table 2 data; an earlier version of this fix did exactly
# that and silently broke both the correct and incorrect test cases.
_ROW_LINE_RE = re.compile(r'^\s*Row \[(.*?)\]:\s*(.*)$')

_QUESTION_STOPWORDS = {
    "how", "what", "which", "who", "whom", "when", "where", "why",
    "is", "are", "was", "were", "does", "did", "do", "the", "a", "an",
}


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


def _numeric_claims_grounded(answer: str, context: str, question: str) -> bool:
    """
    Verify that numbers in the answer are grounded in the context and 
    attributed to the correct entity/field.
    
    2026-08-06 fix (entity-attribution cascade):
    - Extract entities from ACTUAL ROW LABELS in context, not from question
    - For table rows, enforce that number + entity + field all cooccur in 
      the same row's own field-level span
    - For prose, use the weaker sentence-level span check (no per-field split)
    
    This catches the "4.50 for Lychee-FD but you answered it for Moshi" bug 
    by requiring an answer's cited number to appear in a span that ALSO 
    contains the queried entity's row label.
    
    Returns:
        True if all numbers in the answer are properly grounded and attributed
        False if any number is missing from context or attributed to wrong entity
    """
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
                # This ensures "Stop.=570" and "Lat.=826" are separate spans
                fine_spans.extend(
                    f"Row [{row_entity}]: {frag}" for frag in kv_fragments
                )
                continue
            # Row-shaped line but no parseable fields -- treat as plain text
        
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
        
        if not num_is_attributed:
            # Number exists in context but not attached to any queried entity
            # (e.g., "4.50" exists under "Moshi" but question asked about "Lychee-FD")
            return False

    return True


class AnswerAgent(BaseAgent):
    async def _execute(self, state: AgentState) -> AgentState:

        # 2026-07-03 fix: ToolAgent stores web/tool output in
        # state.tool_results, but nothing here ever read it -- the LLM
        # was only ever shown state.retrieved_docs, so a successful web
        # search (e.g. "who won the most recent F1 race?") was silently
        # discarded and the model answered off irrelevant/empty document
        # chunks instead ([ANSWER] logs always said "Using N top
        # documents", never anything about web results, even on
        # web-routed questions where a search had just completed).
        # Build a separate tool-results block and include it whenever
        # present, regardless of whether documents were also retrieved --
        # personal+comparison questions (routing_005-style) need both.
        web_result = state.tool_results.get("web_search") if state.tool_results else None
        tool_context = ""
        web_result_count = 0
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
                tool_context = "\n\n".join(formatted)

        # 2026-07-03 fix: retrieval runs unconditionally upstream of the
        # planner's routing decision, so state.retrieved_docs is always
        # populated -- even for web-only questions where the Planner
        # explicitly decided sources_needed=['web'] and never asked for
        # documents. Previously AnswerAgent ignored that decision and
        # always included top_docs regardless, so a web-only question
        # like "who won the most recent F1 race?" got 5 near-zero-
        # relevance document chunks (e.g. rerank scores ~0.0001, about
        # accessibility/coroutines/mobile AI) mixed into the same
        # context as the real web results. Handing a local LLM unrelated
        # noise next to legitimate facts is a plausible contributor to
        # it blending/contradicting itself (observed: HallucinationMetric
        # flagging the web-grounded answer as disagreeing with context).
        # Respect the router's decision the same way tool_context already
        # does: only include doc context when 'documents' was requested.
        docs_wanted = "documents" in (state.sources_needed or [])
        # Was: top_docs = state.retrieved_docs[:3] if docs_wanted else []
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
            print(f"[ANSWER] No usable content — has_docs={has_any_retrieved_docs}, web_error={bool(web_error)}")
            return state

        # --- Context-window budget check ---------------------------------
        # Everything above builds top_docs/tool_context assuming they'll
        # all fit. Nothing previously checked that against the model's
        # actual context window before sending -- top_docs was capped at
        # 3 items, but web results (tool_context) were never capped at
        # all, and even top_docs alone (3 chunks up to max_tokens_table=
        # 1000 tokens each, per chunker.py, if any hit a table) could
        # exceed num_ctx on its own. Silent overflow means Ollama/Groq
        # truncates the prompt with no signal here that it happened --
        # possibly dropping the actual answer-bearing chunk with no
        # error, no low-confidence flag, nothing.
        #
        # Fix: measure real prompt overhead (the ANSWER_PROMPT template's
        # fixed rules/example text, plus this question) using the same
        # provider-aware counter the chunkers use, reserve room for the
        # model's own output, and only then work out how much budget is
        # actually left for retrieved content. Fill that budget by
        # priority -- protected docs first, then docs by rerank score,
        # then web results -- and cut off (with logging) rather than
        # silently stuffing everything in regardless of size.
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
            # num_ctx is too small even for the prompt scaffold + question
            # + reserved output, before a single piece of retrieved
            # content is added. Not a crash -- fail loudly here so it's
            # visible, then proceed with zero context budget (the LLM
            # will answer from the question alone, same as the "no
            # usable content" path elsewhere in this file).
            print(
                f"[ANSWER][BUDGET] num_ctx={num_ctx} leaves NO room for "
                f"context (scaffold={scaffold_tokens}, reserved_output="
                f"{ANSWER_RESERVED_OUTPUT_TOKENS}, margin="
                f"{BUDGET_SAFETY_MARGIN_TOKENS}). Proceeding with empty "
                f"context -- consider raising num_ctx or lowering "
                f"ANSWER_RESERVED_OUTPUT_TOKENS."
            )
            available_for_context = 0

        # Priority-ordered candidate blocks: protected docs are never
        # dropped by choice (only if the budget can't fit them at all,
        # in which case we log it explicitly rather than quietly
        # exceeding num_ctx); everything else fills remaining budget in
        # the order it was already ranked.
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
                    "kind": "web",
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

        # Keep top_docs/tool_context in sync with what's actually being
        # sent -- state.sources and confidence scoring below both read
        # top_docs, and previously always matched everything built above.
        # If a doc got dropped for budget reasons, it must also disappear
        # from top_docs, or the UI would cite a source that was never
        # actually shown to the LLM.
        top_docs = [b["doc"] for b in included_blocks if b["kind"] == "doc"]
        doc_context = "\n\n".join(b["text"] for b in included_blocks if b["kind"] == "doc")
        tool_context = "\n\n".join(b["text"] for b in included_blocks if b["kind"] == "web")

        context_parts = [p for p in (doc_context, tool_context) if p]
        context = "\n\n".join(context_parts) if context_parts else "(no context available)"

        print(
            f"[ANSWER] Using {len(top_docs)} top documents"
            + (f" + {web_result_count} web results" if tool_context else "")
            + f" | budget: scaffold={scaffold_tokens} used={used_tokens}/"
            f"{available_for_context} num_ctx={num_ctx}"
        )

        try:
            prompt = ANSWER_PROMPT.format(
                question=state.question,
                context=context
            )

            response = await self.call_llm(prompt, max_tokens=ANSWER_RESERVED_OUTPUT_TOKENS)
            state.answer = response.strip()

            # 2026-08-06 fix: NEW GROUNDING LOGIC
            # Pass state.question through so the grounding check can:
            # 1. Extract entities from actual ROW LABELS (not question)
            # 2. Anchor numbers to the specific row AND field they belong to
            # 3. Catch cross-entity attribution (e.g., Moshi's value for Lychee-FD)
            #
            # This replaces the old question-based entity extraction which:
            # - Missed entity aliases ("Lychee-FD (Ours)" treated as 2 entities)
            # - Failed on pronouns ("What does it achieve?" has no entity)
            # - Couldn't discriminate between same entity's different fields
            declined_on_ungrounded_number = not _numeric_claims_grounded(
                state.answer, context, state.question
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

            # NOTE: sources built from top_docs (the docs actually sent to
            # the LLM), not all retrieved_docs - previously these could
            # diverge if Retriever's top_k ever changed independently of
            # AnswerAgent's top-6 slice.
            #
            # filename comes from RetrieverAgent's batched Mongo lookup
            # (attached as doc['filename']). Falls back to a generic
            # "Source N" label only if the lookup didn't run (e.g. db
            # wasn't wired through) or the doc_id had no matching record.
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

            # Surface web results as sources too, so citations/UI reflect
            # what actually informed the answer -- previously these were
            # used in the prompt (once the fix above wired them in) but
            # never appeared in state.sources, making web-grounded answers
            # look document-only to anything consuming state.sources.
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

            # Confidence fix: combined_score is the RRF fusion score, which
            # is rank-based and deliberately tiny/tightly-clustered
            # (RRF_K=60 means scores live in roughly the 0.01-0.05 range
            # no matter how good or bad retrieval actually is). Averaging
            # that directly into a 0-1 confidence score mathematically
            # guarantees confidence_final lands near ~0.42-0.45 for EVERY
            # query regardless of answer quality - which is exactly the
            # symptom observed (confidence stuck at 0.44-0.47 across
            # multiple different, correct answers).
            #
            # rerank_score (the BGE cross-encoder score) carries real
            # signal about retrieval relevance and should be used instead.
            # Falls back to combined_score only if reranking didn't run.
            #
            # 2026-07-03: use top_docs (the docs actually sent to the LLM)
            # rather than all of state.retrieved_docs -- now that doc
            # context is excluded entirely for web-only questions, using
            # the full retrieved_docs list here would still drag
            # confidence down using near-zero scores from chunks that
            # were never shown to the model. When top_docs is empty
            # (web-only), avg_doc_score is 0 and confidence rests solely
            # on the planner's confidence, which is correct since there's
            # no retrieval signal to speak of.
            scored_docs = [
                doc.get("rerank_score", doc.get("combined_score", 0.0))
                for doc in top_docs
            ]
            avg_doc_score = sum(scored_docs) / len(scored_docs) if scored_docs else 0.0

            # rerank_score from a cross-encoder isn't naturally bounded to
            # [0, 1] - clamp defensively so confidence_final stays sane
            # even if the model's raw score is outside that range.
            avg_doc_score = max(0.0, min(avg_doc_score, 1.0))

            state.confidence_final = min(
                state.confidence * 0.5 + avg_doc_score * 0.5, 1.0
            )
            if declined_on_ungrounded_number:
                state.confidence_final = min(state.confidence_final, 0.3)

            print(f"[ANSWER] Generated response with {len(state.sources)} sources")
            print(f"[ANSWER] Confidence: {state.confidence_final:.2f}")

        except Exception as e:
            # 2026-07-03: this was swallowing the real cause entirely --
            # state.error was set but nothing printed it, so a failure
            # here (e.g. prompt exceeding the model's context window
            # after web results were added to context, a malformed
            # ANSWER_PROMPT.format() call, or an LLM/connection error)
            # was indistinguishable from any other failure. Print it so
            # it's visible in eval logs instead of just "Sorry, I
            # couldn't generate an answer" with no explanation.
            print(f"[ANSWER] Answer generation FAILED: {type(e).__name__}: {e}")
            state.error = f"Answer agent error: {str(e)}"
            state.answer = "Sorry, I couldn't generate an answer."
            state.sources = []
            state.confidence_final = 0.0

        return state