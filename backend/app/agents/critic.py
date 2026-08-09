import re
import json
from app.agents.base import BaseAgent
from app.agents.state import AgentState, _compute_hash
from app.agents.prompts import CRITIC_PROMPT

# --------------------------------------------------------------------------
# Critic Agent with failure_type classification
# --------------------------------------------------------------------------
_FAILED_ANSWER = "Sorry, I couldn't generate an answer."

# Threshold for "strong grounding" — if >= this, facts in answer are grounded
_GROUNDING_THRESHOLD_STRONG = 0.8
_GROUNDING_THRESHOLD_WEAK = 0.2

# Threshold for retrieval confidence — if < this, retrieval likely failed
_RETRIEVAL_CONFIDENCE_THRESHOLD = 0.7


class CriticAgent(BaseAgent):
    """
    Critic Agent: Validates answer quality and classifies failure type.

    2026-08-09 FIX: Now validates against ACTUAL EVIDENCE USED by AnswerAgent,
    not just state.retrieved_docs. This includes documents, web, calculator,
    weather, and other tool results.

    Failure classification considers sources_needed:
    - "generation": Good evidence + bad answer
    - "retrieval": Missing/wrong evidence (routed source had no results)
    - "planning": Correct source exists but wrong source was selected
    - "tool": Tool execution failed
    - "unknown": Can't confidently classify
    
    STRATEGY: Use deterministic rules FIRST (faster, more reliable),
    only LLM-classify when deterministic signals are ambiguous.
    """

    # ---- Grounding backstop -------------------------------------------

    _NUMBER_RE = re.compile(r'\d[\d,]*(?:\.\d+)?%?')
    _PROPER_NOUN_RE = re.compile(r'\b[A-Z][A-Za-z0-9\-]{2,}\b')

    def _extract_checkable_facts(self, answer: str) -> list[str]:
        """
        Pulls out concrete, checkable tokens from the answer: numbers
        (with optional decimal/percent) and capitalized words (proper
        nouns, acronyms, tool/library names -- the terms most likely to
        indicate a real vs. hallucinated claim). Deliberately coarse --
        this is a backstop signal, not a full fact-verification system.
        """
        numbers = self._NUMBER_RE.findall(answer)
        proper_nouns = self._PROPER_NOUN_RE.findall(answer)

        # Drop common sentence-starter words that happen to be capitalized
        # but aren't meaningful entities (reduces false "ungrounded" hits).
        stopwords = {"The", "This", "That", "These", "Those", "According",
                     "Based", "In", "It", "There", "Answer", "Source"}
        proper_nouns = [w for w in proper_nouns if w not in stopwords]

        return list(set(numbers + proper_nouns))

    def _compute_grounding_score(self, answer: str, context: str) -> float:
        """
        Fraction of checkable facts in the answer that appear (case-
        insensitive) somewhere in the retrieved context. Returns 1.0
        when there are no checkable facts to verify (e.g. a purely
        qualitative answer) -- absence of evidence isn't evidence of
        ungroundedness here, so we don't penalize what we can't check.
        """
        facts = self._extract_checkable_facts(answer)
        if not facts:
            return 1.0

        context_lower = context.lower()
        matched = sum(1 for f in facts if f.lower() in context_lower)
        return matched / len(facts)

    def _build_actual_evidence_context(self, state: AgentState) -> str:
        """
        Build context from the ACTUAL EVIDENCE that AnswerAgent used,
        not just state.retrieved_docs. This includes:
        - Documents (if "documents" was routed)
        - Web results (if "web" was routed)
        - Calculator results (if "calculator" was routed)
        - Weather/Slack/Email (if "tool" was routed)
        - Database results (if "database" was routed)
        - Metadata (if available)
        
        This ensures Critic evaluates against what the LLM actually saw.
        """
        sources_needed = state.sources_needed or []
        context_parts = []

        # ── Documents ────────────────────────────────────────────────
        if "documents" in sources_needed and state.retrieved_docs:
            for i, doc in enumerate(state.retrieved_docs, 1):
                context_parts.append(f"[Source {i}] {doc['text']}")

        # ── Metadata ─────────────────────────────────────────────────
        if state.metadata_answer:
            meta_text = "\n".join(
                f"{k}: {v}" for k, v in state.metadata_answer.items()
            )
            if meta_text:
                context_parts.append(f"[Metadata]\n{meta_text}")

        # ── Web Results ──────────────────────────────────────────────
        if "web" in sources_needed:
            web_result = state.tool_results.get("web_search") if state.tool_results else None
            if web_result and "error" not in web_result:
                entries = web_result.get("results", [])
                for i, entry in enumerate(entries, 1):
                    title = entry.get("title", "")
                    snippet = entry.get("snippet") or entry.get("content") or ""
                    context_parts.append(f"[Web {i}] {title}\n{snippet}")

        # ── Calculator ───────────────────────────────────────────────
        if "calculator" in sources_needed:
            calc_result = state.tool_results.get("calculator") if state.tool_results else None
            if calc_result and "error" not in calc_result:
                expr = calc_result.get("expression", "")
                result = calc_result.get("result")
                if result is not None:
                    context_parts.append(f"[Calculator Result] {expr} = {result}")

        # ── Tool Results (Weather, Slack, Email) ─────────────────────
        if "tool" in sources_needed:
            tool_results = state.tool_results or {}
            for kind in ("weather", "slack", "email"):
                tool_result = tool_results.get(kind)
                if tool_result and "error" not in tool_result:
                    if kind == "weather":
                        temp = tool_result.get("temperature")
                        desc = tool_result.get("description", "")
                        loc = tool_result.get("location", "the requested location")
                        if temp is not None or desc:
                            context_parts.append(
                                f"[Weather Result] {loc}: {temp}°C, {desc}".strip()
                            )
                    elif kind == "slack":
                        channel = tool_result.get("channel", "the requested channel")
                        context_parts.append(f"[Slack Result] Message was posted to {channel}")
                    elif kind == "email":
                        to_email = tool_result.get("to_email", "the requested recipient")
                        context_parts.append(f"[Email Result] Email was sent to {to_email}")

        # ── Database Results ─────────────────────────────────────────
        if "database" in sources_needed:
            db_result = state.tool_results.get("database") if state.tool_results else None
            if db_result and "error" not in db_result:
                context_parts.append(f"[Database Result] {db_result}")

        return "\n\n".join(context_parts) if context_parts else "(no evidence available)"

    # ---- Deterministic failure classification -------------------------

    def _classify_failure_deterministic(self, state: AgentState, grounding_score: float,
                                        evidence_context: str) -> str:
        """
        Attempt to classify failure type using deterministic rules.
        
        Returns the classified failure_type ("generation", "retrieval", 
        "planning", "tool", or None if ambiguous).
        
        IMPORTANT: Only called when is_valid=False (answer failed validation).
        Must consider sources_needed to avoid false "retrieval" classifications
        for questions that never requested documents.
        """
        sources_needed = state.sources_needed or []

        # RULE 1: Tool was explicitly executed but tool_results has error
        for source in sources_needed:
            if source in ("calculator", "weather", "slack", "email"):
                result = state.tool_results.get(source) if state.tool_results else None
                if result and result.get("error"):
                    print(f"[CRITIC] Deterministic: tool '{source}' has error → failure_type=tool")
                    return "tool"

        # RULE 2: Documents were routed but not retrieved
        if "documents" in sources_needed and not state.retrieved_docs:
            print("[CRITIC] Deterministic: documents routed but not retrieved → failure_type=retrieval")
            return "retrieval"

        # RULE 3: Web was routed but web search had no results
        if "web" in sources_needed:
            web_result = state.tool_results.get("web_search") if state.tool_results else None
            if not web_result or web_result.get("error") or not web_result.get("results"):
                print("[CRITIC] Deterministic: web routed but no web results → failure_type=tool")
                return "tool"

        # RULE 4: Strong grounding (facts ARE in evidence) but answer
        #         marked invalid → judge error or answer structure issue,
        #         likely not a generation hallucination.
        if grounding_score >= _GROUNDING_THRESHOLD_STRONG:
            print(f"[CRITIC] Deterministic: strong grounding ({grounding_score:.2f}) "
                  f"despite invalid → ambiguous, need LLM classification")
            return None

        # RULE 5: Zero grounding (facts NOT in evidence) AND evidence is
        #         non-empty → generation hallucination
        if grounding_score < _GROUNDING_THRESHOLD_WEAK and evidence_context != "(no evidence available)":
            print(f"[CRITIC] Deterministic: near-zero grounding ({grounding_score:.2f}) "
                  f"despite evidence present → failure_type=generation")
            return "generation"

        # RULE 6: No evidence at all (no docs, no web, no tool results)
        #         AND something was routed to → retrieval/tool failure
        if evidence_context == "(no evidence available)":
            if "documents" in sources_needed:
                print("[CRITIC] Deterministic: no evidence despite documents routed → failure_type=retrieval")
                return "retrieval"
            if any(s in sources_needed for s in ("web", "calculator", "tool")):
                print("[CRITIC] Deterministic: no evidence despite tool routed → failure_type=tool")
                return "tool"

        # RULE 7: Moderate grounding + evidence present → ambiguous
        print("[CRITIC] Deterministic: grounding/evidence ambiguous → need LLM classification")
        return None

    # ---- JSON extraction (unchanged) -----------------------------------

    def _extract_json(self, text: str) -> dict:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        candidate = self._extract_balanced_json(text)
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        print(f"[CRITIC] All JSON extraction strategies failed. Raw: {text[:300]!r}")
        return {
            "valid": False,
            "confidence": 0,
            "failure_type": "unknown",
            "issues": ["Could not parse critic response"],
            "needs_more_info": True
        }

    async def _execute(self, state: AgentState) -> AgentState:
        """Validate the answer, set critic_confidence, and classify failure_type."""

        if not state.answer or state.answer.strip() == "Searching for relevant information...":
            state.error = "CriticAgent called before AnswerAgent produced an answer"
            state.is_valid = False
            state.failure_type = "unknown"
            print("[CRITIC] Skipped — no answer to validate")
            return state

        if state.error or state.answer.strip() == _FAILED_ANSWER:
            state.is_valid = False
            state.failure_type = "generation"  # AnswerAgent failed to generate
            state.validation_issues = ["Answer generation failed upstream; not evaluated"]
            state.critic_confidence = 0.0
            state.confidence_final = 0.0
            print(
                f"[CRITIC] Skipped — answer generation failed upstream "
                f"(state.error={state.error!r}), failure_type=generation, forcing confidence_final=0.0"
            )
            return state

        # 2026-08-09 FIX: Build context from ACTUAL EVIDENCE used by AnswerAgent
        evidence_context = self._build_actual_evidence_context(state)

        # For LLM evaluation, include truncated evidence
        truncated_context = "\n".join([
            line[:200] for line in evidence_context.split("\n")
        ])

        prompt = CRITIC_PROMPT.format(
            question=state.question,
            context=truncated_context,
            answer=state.answer
        )

        try:
            response = await self.call_llm(prompt)
        except Exception as e:
            state.error = f"Critic LLM call failed: {str(e)}"
            state.is_valid = False
            state.failure_type = "unknown"
            print(f"[CRITIC] LLM call failed: {e}")
            return state

        criticism = self._extract_json(response)

        state.is_valid = bool(criticism.get("valid", False))
        state.validation_issues = criticism.get("issues", [])

        raw_conf = criticism.get("confidence", 0)
        if raw_conf is None:
            raw_conf = 0
        raw_conf = float(raw_conf)
        critic_conf = raw_conf / 100.0 if raw_conf > 1.0 else raw_conf
        state.critic_confidence = max(0.0, min(1.0, critic_conf))

        # ── Grounding check against ACTUAL EVIDENCE ──────────────────
        grounding_score = self._compute_grounding_score(state.answer, evidence_context)

        # Retrieval score: average of actual evidence scores (not just docs)
        retrieval_score = 0.5  # default neutral
        if state.retrieved_docs:
            top_doc_score = state.retrieved_docs[0].get("rerank_score", 0.5)
            if top_doc_score is not None:
                retrieval_score = max(0.0, min(1.0, float(top_doc_score)))

        unexplained_rejection = (
            not state.is_valid
            and state.critic_confidence == 0.0
            and not state.validation_issues
        )

        if unexplained_rejection and grounding_score >= 0.8 and retrieval_score >= 0.7:
            print(
                f"[CRITIC] Overriding unexplained 0-confidence rejection "
                f"(grounding_score={grounding_score:.2f}, retrieval_score={retrieval_score:.2f}, "
                f"no issues stated) — treating as valid"
            )
            state.is_valid = True
            state.failure_type = ""  # Answer is now valid; no failure to classify
            state.critic_confidence = max(state.critic_confidence, 0.75)
            state.validation_issues = []

        # Overconfident acceptance: high confidence despite near-zero grounding
        overconfident_acceptance = (
            state.is_valid
            and state.critic_confidence >= 0.8
            and grounding_score < 0.2
        )

        if overconfident_acceptance:
            print(
                f"[CRITIC] Overriding high-confidence acceptance despite "
                f"near-zero grounding (grounding_score={grounding_score:.2f}, "
                f"critic_confidence={state.critic_confidence:.2f}) "
                f"— judge likely rubber-stamped a hallucination"
            )
            state.is_valid = False
            state.failure_type = "generation"  # Judge missed a hallucination
            state.critic_confidence = min(state.critic_confidence, 0.2)
            state.validation_issues = state.validation_issues + [
                f"Deterministic grounding check found near-zero overlap "
                f"(score={grounding_score:.2f}) between answer claims and "
                f"available evidence, despite judge approval — treating as "
                f"likely hallucination"
            ]

        # ── Failure type classification ───────────────────────────────────
        # Only classify if is_valid=False (i.e., answer failed validation).
        # If the answer passed, there's no failure to classify.
        if not state.is_valid and not state.failure_type:
            # Try deterministic classification first (faster, more reliable)
            deterministic_type = self._classify_failure_deterministic(
                state, grounding_score, evidence_context
            )

            if deterministic_type:
                state.failure_type = deterministic_type
            else:
                # Deterministic rules were ambiguous; use LLM classification
                # (which already ran above, so extract from existing response)
                llm_type = criticism.get("failure_type", "unknown")
                if llm_type not in ("generation", "retrieval", "planning", "tool"):
                    llm_type = "unknown"
                state.failure_type = llm_type
                print(f"[CRITIC] LLM-classified failure type: {llm_type}")

        state.confidence_final = round(
            0.7 * state.critic_confidence + 0.3 * retrieval_score, 4
        )

        # Update change detection hashes (for early stop logic in graph routing)
        state.last_answer_hash = _compute_hash(state.answer)

        print(f"[CRITIC] Valid: {state.is_valid}")
        print(f"[CRITIC] Failure type: {state.failure_type or '(none)'}")
        print(f"[CRITIC] Critic confidence: {state.critic_confidence:.2f}")
        print(f"[CRITIC] Grounding score:   {grounding_score:.2f}")
        print(f"[CRITIC] Retrieval score:   {retrieval_score:.4f}")
        print(f"[CRITIC] Evidence sources:  {state.sources_needed or []}")
        print(f"[CRITIC] confidence_final:  {state.confidence_final:.4f}")
        if state.validation_issues:
            print(f"[CRITIC] Issues: {state.validation_issues}")

        return state