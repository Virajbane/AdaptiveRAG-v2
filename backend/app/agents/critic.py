import re
import json
from app.agents.base import BaseAgent
from app.agents.state import AgentState
from app.agents.prompts import CRITIC_PROMPT

# --------------------------------------------------------------------------
# 2026-07-04 bug: Critic scoring a failed generation as if it were valid
# (existing comment block unchanged, see below)
# --------------------------------------------------------------------------
_FAILED_ANSWER = "Sorry, I couldn't generate an answer."


class CriticAgent(BaseAgent):
    """
    Critic Agent: Validates answer quality.

    - Checks for hallucinations
    - Verifies evidence grounding
    - Detects missing info
    - Returns confidence score that feeds into confidence_final

    2026-07-05 fix — unexplained 0-confidence rejections:
      Recurring pattern across multiple sessions (Q6, Q10, Q9, Q15, Q18):
      the judge LLM returns valid=False, confidence=0, issues=[] (empty)
      for answers that are demonstrably well-grounded and correct --
      confirmed on Q18 where retrieval score was 0.97 and the answer text
      matched the source almost verbatim, yet the judge rejected it twice
      with zero confidence and zero stated reason, exhausting retries and
      returning a correct answer mislabeled at 29% confidence.

      A 0-confidence rejection with NO issues listed is qualitatively
      different from a rejection that names a specific problem (e.g. the
      ARI/clustering case elsewhere in testing, which correctly flagged a
      real gap) -- an unexplained blanket rejection looks like judge
      noise, not a genuine catch.

      Fix: compute a deterministic grounding score (do the answer's
      concrete facts -- numbers, percentages, proper nouns -- actually
      appear in the retrieved context?) as a backstop. If the judge's
      rejection is unexplained (empty issues) AND grounding is strong,
      override the rejection instead of trusting an unexplained verdict.
      This does NOT touch explained rejections (non-empty issues) --
      those may be catching something real and are left as-is.

    2026-07-XX fix — override was firing on weak retrieval:
      The override above was justified using Q18, where retrieval_score
      was 0.97 -- i.e. it was only ever meant to apply when BOTH grounding
      AND retrieval agree the answer is solid. Eval surfaced a case
      (decline_01 rerun on the Lychee-FD paper, retrieval top_score=0.5493)
      where grounding_score alone (0.83) triggered the override despite much
      weaker retrieval, overturning a judge rejection that had correctly
      caught a fabricated claim ("Lychee-FD achieves a 28.5% gain on
      FullDuplexBench 1.5" -- not a real figure from the source, just a
      topically-adjacent number). Now requires retrieval_score >= 0.7 too:
      weak retrieval + an unexplained rejection is more likely a genuine
      catch, not judge noise, so we no longer override in that regime.

    2026-07-XX fix — overconfident_acceptance was unreachable:
      This check (added 2026-07-10, see below) was nested inside the
      unexplained_rejection block, which only ever runs when
      grounding_score >= 0.8 -- but overconfident_acceptance requires
      grounding_score < 0.2. Those two conditions can never both be true,
      so the check could never fire. Moved out to run independently on
      every judge acceptance, which is what its own docstring describes.
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
        return {"valid": False, "confidence": 0, "issues": ["Could not parse critic response"], "needs_more_info": True}

    async def _execute(self, state: AgentState) -> AgentState:
        """Validate the answer and set critic_confidence + confidence_final."""

        if not state.answer or state.answer.strip() == "Searching for relevant information...":
            state.error = "CriticAgent called before AnswerAgent produced an answer"
            state.is_valid = False
            print("[CRITIC] Skipped — no answer to validate")
            return state

        if state.error or state.answer.strip() == _FAILED_ANSWER:
            state.is_valid = False
            state.validation_issues = ["Answer generation failed upstream; not evaluated"]
            state.critic_confidence = 0.0
            state.confidence_final = 0.0
            print(
                f"[CRITIC] Skipped — answer generation failed upstream "
                f"(state.error={state.error!r}), forcing confidence_final=0.0"
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

        # retrieval_score computed here now (moved up from below) so both
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
            # Don't just set 1.0 -- reflect that this came from the
            # deterministic backstop, not genuine LLM confidence, so it's
            # visibly distinguishable in logs/metrics from a normal pass.
            state.critic_confidence = max(state.critic_confidence, 0.75)
            state.validation_issues = []

        # 2026-07-10 fix — unexplained high-confidence ACCEPTANCE despite
        # near-zero grounding:
        #   Mirror image of the override above. That fix protects against
        #   the judge saying "invalid" when the answer is actually fine.
        #   This protects the opposite and more dangerous direction: the
        #   judge saying "valid, high confidence" while the deterministic
        #   grounding check finds essentially no overlap between the
        #   answer's claims and the retrieved context -- i.e. a likely
        #   hallucination that the judge rubber-stamped.
        #
        #   Caught directly in eval: golden-set question about a benchmark
        #   never mentioned in the source document (SQuAD) still produced
        #   a confident-sounding accuracy figure. CriticAgent's own
        #   grounding_score computed 0.00 (zero of the answer's checkable
        #   facts appeared anywhere in retrieved context), yet the judge
        #   returned valid=True, confidence=1.00, and confidence_final
        #   still landed at 0.99 -- the highest-confidence answer in that
        #   entire eval run, despite being the one most likely fabricated.
        #
        #   2026-07-XX: un-nested this from the unexplained_rejection block
        #   above, where it was structurally unreachable (that block only
        #   runs when grounding_score >= 0.8, but this needs grounding_score
        #   < 0.2 -- the two can never both be true). Now runs independently
        #   on every judge acceptance, explained or not.
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
            state.critic_confidence = min(state.critic_confidence, 0.2)
            state.validation_issues = state.validation_issues + [
                f"Deterministic grounding check found near-zero overlap "
                f"(score={grounding_score:.2f}) between answer claims and "
                f"retrieved context, despite judge approval — treating as "
                f"likely hallucination"
            ]

        state.confidence_final = round(
            0.7 * state.critic_confidence + 0.3 * retrieval_score, 4
        )

        print(f"[CRITIC] Valid: {state.is_valid}")
        print(f"[CRITIC] Critic confidence: {state.critic_confidence:.2f}")
        print(f"[CRITIC] Grounding score:   {grounding_score:.2f}")
        print(f"[CRITIC] Retrieval score:   {retrieval_score:.4f}")
        print(f"[CRITIC] confidence_final:  {state.confidence_final:.4f}")
        if state.validation_issues:
            print(f"[CRITIC] Issues: {state.validation_issues}")

        return state