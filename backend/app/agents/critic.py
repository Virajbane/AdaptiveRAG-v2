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

    2026-07-XX: Enhanced to produce failure_type classification that
    determines which component to retry:
    
    - "generation": AnswerAgent hallucinated or failed synthesis
    - "retrieval": Retriever returned wrong/insufficient chunks
    - "planning": Planner misrouted (web vs docs, wrong sources)
    - "tool": ToolAgent execution failed
    - "unknown": Can't confidently classify (return as-is, no retry)
    
    STRATEGY: Use deterministic rules FIRST, only LLM-classify if
    deterministic signals are ambiguous. This reduces token usage and
    avoids LLM noise in classification.
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

    # ---- Deterministic failure classification -------------------------

    def _classify_failure_deterministic(self, state: AgentState, grounding_score: float, 
                                        retrieval_score: float) -> str:
        """
        Attempt to classify failure type using deterministic rules.
        
        Returns the classified failure_type ("generation", "retrieval", 
        "planning", "tool", or None if ambiguous — meaning LLM will classify).
        
        IMPORTANT: This runs ONLY when is_valid=False. For acceptances,
        we use the overconfident_acceptance check and don't need to classify.
        """

        # RULE 1: No documents retrieved at all → retrieval failure
        if not state.retrieved_docs and "documents" in state.sources_needed:
            return "retrieval"

        # RULE 2: Retrieval score is very low despite docs being present
        #         (could be reranking failure or semantic mismatch)
        if retrieval_score < 0.5:  # very weak signal
            return "retrieval"

        # RULE 3: Tool was requested and tool execution error was recorded
        if "tools" in state.sources_needed and state.tool_results.get("error"):
            return "tool"

        # RULE 4: Planner error was recorded (JSON parse failure, etc.)
        if state.error and "planner" in state.error.lower():
            return "planning"

        # RULE 5: Strong grounding (facts ARE in context) but answer still
        #         marked invalid → likely a judge error, not a generation error.
        #         This is ambiguous without more info; LLM should classify.
        if grounding_score >= _GROUNDING_THRESHOLD_STRONG:
            return None  # ambiguous, let LLM decide

        # RULE 6: Zero grounding (facts NOT in context) AND retrieval was
        #         strong → generation failure (hallucination)
        if grounding_score < _GROUNDING_THRESHOLD_WEAK and retrieval_score >= _RETRIEVAL_CONFIDENCE_THRESHOLD:
            return "generation"

        # RULE 7: Weak grounding AND weak retrieval → could be retrieval
        if grounding_score < _GROUNDING_THRESHOLD_WEAK and retrieval_score < _RETRIEVAL_CONFIDENCE_THRESHOLD:
            return "retrieval"

        # RULE 8: Moderate grounding (neither strong nor weak) → ambiguous
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

        context = "\n".join([
            f"[{i}] {doc['text'][:200]}..."
            for i, doc in enumerate(state.retrieved_docs, 1)
        ])

        prompt = CRITIC_PROMPT.format(
            question=state.question,
            context=context,
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

        # ── Grounding backstop ────────────────────────────────────────
        # Full context (not the 200-char-truncated preview built above)
        # so number/entity matching isn't penalized by arbitrary truncation.
        full_context = "\n".join(doc["text"] for doc in state.retrieved_docs)
        grounding_score = self._compute_grounding_score(state.answer, full_context)

        # retrieval_score computed here (moved up from below) so both
        # override checks below can weigh it alongside grounding_score.
        top_doc_score = None
        if state.retrieved_docs:
            top_doc_score = state.retrieved_docs[0].get("rerank_score", 0.5)
        if top_doc_score is None:
            top_doc_score = 0.5
        retrieval_score = float(top_doc_score)
        retrieval_score = max(0.0, min(1.0, retrieval_score))

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

        # 2026-07-10 fix — unexplained high-confidence ACCEPTANCE despite
        # near-zero grounding (mirrored from unexplained_rejection above).
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
                f"retrieved context, despite judge approval — treating as "
                f"likely hallucination"
            ]

        # ── Failure type classification ───────────────────────────────────
        # Only classify if is_valid=False (i.e., answer failed validation).
        # If the answer passed, there's no failure to classify.
        if not state.is_valid and not state.failure_type:
            # Try deterministic classification first
            deterministic_type = self._classify_failure_deterministic(
                state, grounding_score, retrieval_score
            )

            if deterministic_type:
                state.failure_type = deterministic_type
                print(f"[CRITIC] Deterministic failure classification: {deterministic_type}")
            else:
                # Deterministic rules were ambiguous; fall back to asking the LLM
                # (which already ran above, so we can extract from existing response)
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
        print(f"[CRITIC] confidence_final:  {state.confidence_final:.4f}")
        if state.validation_issues:
            print(f"[CRITIC] Issues: {state.validation_issues}")

        return state