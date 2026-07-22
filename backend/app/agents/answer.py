from app.agents.base import BaseAgent
from app.agents.state import AgentState
from app.agents.prompts import ANSWER_PROMPT
import re

# Excludes digits embedded inside hyphenated/alphanumeric identifiers
# (e.g. "StepAudio-2-mini", "GPT-5-Duplex") -- those aren't factual
# claims to verify, they're part of a proper noun. Confirmed root cause
# of a false-decline: "StepAudio-2-mini" (a correct answer) was rejected
# because the embedded "2" was treated as an unverified numeric claim.
_NUM_RE = re.compile(r'(?<![A-Za-z0-9-])\d+(?:\.\d+)?%?(?![A-Za-z0-9-])')


def _numeric_claims_grounded(answer: str, context: str, question: str) -> bool:
    """If a number in the answer never appears anywhere in the context
    actually handed to the LLM, it didn't come from that context --
    it's either training-data recall or invention.

    2026-07-14 fix: the original version only checked substring presence
    of the number anywhere in context ("num in context"). That's too
    weak -- it can't tell WHICH entity a number is attached to. Confirmed
    root cause of the UTMOS eval failures (answer_utmos_freezeomni_1,
    answer_utmos_moshi_1): both wrong answers reused "4.50" -- which IS
    present in context (it's Lychee-FD's own score sitting a few lines
    away) -- so the old check passed even though "4.50" was never
    actually paired with "Freeze-Omni" or "Moshi" anywhere in the
    source text. The guard was checking existence, not attribution.

    Fix: split context into line/sentence-level spans, and require that
    for each number in the answer, at least one context span contains
    BOTH that number AND an entity mentioned in the question. This is a
    crude string-proximity check (not real entity linking/NER), but it
    directly catches the observed failure mode: a number that's real but
    borrowed from a different entity's row/sentence.

    If the question doesn't contain any capitalized entity-like tokens
    (e.g. a purely conceptual question with no named entity), we fall
    back to the original "does the number appear anywhere" check, since
    there's no entity to anchor against.
    """
    numbers_in_answer = _NUM_RE.findall(answer)
    if not numbers_in_answer:
        return True

    # crude entity extraction: capitalized multi-char tokens from the
    # question (catches things like "Freeze-Omni", "Moshi", "Lychee-FD",
    # "Table 1"). Not real NER, but cheap and matches the report's
    # observed failure cases. Filter out common sentence-initial question
    # words that get capitalized purely by position, not because they're
    # proper nouns -- otherwise a plain question like "How many layers
    # does the model use?" gets "How" treated as an entity, and since
    # "how" never appears in context, every number in the answer would
    # be wrongly declined.
    _QUESTION_STOPWORDS = {
        "how", "what", "which", "who", "whom", "when", "where", "why",
        "is", "are", "was", "were", "does", "did", "do", "the", "a", "an",
    }
    entities = [
        e for e in re.findall(r"\b[A-Z][A-Za-z0-9\-]{2,}\b", question)
        if e.lower() not in _QUESTION_STOPWORDS
    ]

    if not entities:
        # No named entity to anchor against -- fall back to the
        # original, weaker existence check rather than false-declining
        # every number-containing answer to a non-comparison question.
        return all(num in context for num in numbers_in_answer)

    # Split context into sentence/line-level spans so we can check
    # whether a number and an entity actually co-occur in the SAME
    # span, rather than just both existing somewhere in the whole blob.
    # Split on '.' only when NOT between two digits, so decimal numbers
    # like "4.21" don't get fractured into separate spans.
    context_spans = re.split(r"(?<!\d)\.(?!\d)|\n", context)

    # Drop entities that appear in EVERY numbered span (e.g. a metric
    # name like "UTMOS" mentioned in every row, or a "[Source N]" label
    # digit that isn't a real fact). Those don't discriminate between
    # rows, so matching against them defeats the point of the check --
    # keep only entities that appear in a strict subset of spans, i.e.
    # ones that actually distinguish which row a number belongs to.
    spans_with_numbers = [
        s for s in context_spans
        if re.search(r"\d", s) and not re.match(r"^\s*\[.*\]\s*$", s)
    ]
    discriminating = [
        e for e in entities
        if 0 < sum(1 for s in spans_with_numbers if e.lower() in s.lower()) < len(spans_with_numbers)
    ]
    entities = discriminating or entities

    for num in numbers_in_answer:
        num_and_entity_cooccur = any(
            num in span and any(e.lower() in span.lower() for e in entities)
            for span in context_spans
        )
        if not num_and_entity_cooccur:
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

        doc_context = "\n\n".join([
            f"[Source {i}]\n{doc['text']}"
            for i, doc in enumerate(top_docs, 1)
        ])

        context_parts = [p for p in (doc_context, tool_context) if p]
        context = "\n\n".join(context_parts) if context_parts else "(no context available)"

        print(
            f"[ANSWER] Using {len(top_docs)} top documents"
            + (f" + {web_result_count} web results" if tool_context else "")
        )

        try:
            prompt = ANSWER_PROMPT.format(
                question=state.question,
                context=context
            )

            response = await self.call_llm(prompt)
            state.answer = response.strip()

            # 2026-07-14 fix: pass state.question through so the grounding
            # check can anchor numbers to the entity actually being asked
            # about, not just check raw existence anywhere in context.
            # See _numeric_claims_grounded docstring for the UTMOS
            # entity-attribution bug this closes.
            declined_on_ungrounded_number = not _numeric_claims_grounded(
                state.answer, context, state.question
            )
            if declined_on_ungrounded_number:
                print(
                    f"[ANSWER] Numeric claim not found verbatim in context "
                    f"(or not attributed to the entity asked about) "
                    f"— declining rather than shipping unverified number. "
                    f"Original answer was: {state.answer!r}"
                )
                state.answer = (
                    "I found related content in the document, but couldn't verify "
                    "a specific number for this with confidence from the retrieved "
                    "text. This may be a figure or chart value that wasn't "
                    "extracted as readable text, or the number belongs to a "
                    "different entity than the one asked about."
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