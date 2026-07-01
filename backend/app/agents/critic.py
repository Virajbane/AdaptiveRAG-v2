import re
import json
from app.agents.base import BaseAgent
from app.agents.state import AgentState
from app.agents.prompts import CRITIC_PROMPT


class CriticAgent(BaseAgent):
    """
    Critic Agent: Validates answer quality.

    - Checks for hallucinations
    - Verifies evidence grounding
    - Detects missing info
    - Returns confidence score that feeds into confidence_final

    Note: model selection (fast vs. main LLM) is handled by
    AgentOrchestrator, which injects the right LLMProvider instance.
    This class does not need its own __init__.
    """

    def _extract_json(self, text: str) -> dict:
        """
        Robustly extract JSON from model output even when the model
        writes preamble text before the JSON block.

        Strategy:
        1. Try direct parse (model was well-behaved).
        2. Balanced-brace scan for the first complete {...} object
           (handles preamble like "The answer looks correct... { ... }",
           and correctly stops at the end of the FIRST object instead
           of spanning to the last '}' in the text).
        3. Try to extract from a markdown code block.
        4. Give up and return a safe default.
        """
        text = text.strip()

        # 1. Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Balanced-brace scan for the first complete {...} object.
        #    Replaces the old `text[start:rfind('}')]` approach, which
        #    used the LAST '}' in the string. That broke when qwen2.5
        #    emitted two JSON blocks back to back (e.g. one example
        #    block + one real answer) — the old slice spanned across
        #    both blocks and produced invalid JSON. See
        #    BaseAgent._extract_balanced_json for the implementation.
        candidate = self._extract_balanced_json(text)
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # 3. Markdown code block  ```json ... ```
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 4. Safe default — treat as invalid so retry can fire if budget allows
        print(f"[CRITIC] All JSON extraction strategies failed. Raw: {text[:300]!r}")
        return {"valid": False, "confidence": 0, "issues": ["Could not parse critic response"], "needs_more_info": True}

    async def _execute(self, state: AgentState) -> AgentState:
        """Validate the answer and set critic_confidence + confidence_final."""

        # Guard: don't run if AnswerAgent hasn't produced anything yet
        if not state.answer or state.answer.strip() == "Searching for relevant information...":
            state.error = "CriticAgent called before AnswerAgent produced an answer"
            state.is_valid = False
            print("[CRITIC] Skipped — no answer to validate")
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
            # Real LLM / connection failure — distinct from a parse failure.
            state.error = f"Critic LLM call failed: {str(e)}"
            state.is_valid = False
            print(f"[CRITIC] LLM call failed: {e}")
            return state

        criticism = self._extract_json(response)

        state.is_valid = bool(criticism.get("valid", False))
        state.validation_issues = criticism.get("issues", [])

        # Normalise confidence to 0-1 regardless of whether the model
        # returned 0-1 or 0-100 (qwen2.5 does both unpredictably).
        #
        # NOTE: dict.get(key, default) only falls back to `default` when
        # the key is MISSING. If the model emits `"confidence": null`
        # literally, the key IS present with value None, so .get() still
        # returns None and float(None) crashes. Guard explicitly.
        raw_conf = criticism.get("confidence", 0)
        if raw_conf is None:
            raw_conf = 0
        raw_conf = float(raw_conf)
        critic_conf = raw_conf / 100.0 if raw_conf > 1.0 else raw_conf
        state.critic_confidence = max(0.0, min(1.0, critic_conf))

        # ── Blend into confidence_final ───────────────────────────────
        # Formula: 70 % critic judgment + 30 % retrieval rerank score.
        # This ensures a good answer from strong evidence scores higher
        # than a good answer from weak evidence.
        #
        # Same None-guard as above — a chunk dict can carry the key
        # "rerank_score" with value None (e.g. reranker partially failed
        # on one item) rather than omitting the key entirely.
        top_doc_score = None
        if state.retrieved_docs:
            top_doc_score = state.retrieved_docs[0].get("rerank_score", 0.5)
        if top_doc_score is None:
            top_doc_score = 0.5
        retrieval_score = float(top_doc_score)
        retrieval_score = max(0.0, min(1.0, retrieval_score))

        state.confidence_final = round(
            0.7 * state.critic_confidence + 0.3 * retrieval_score, 4
        )

        print(f"[CRITIC] Valid: {state.is_valid}")
        print(f"[CRITIC] Critic confidence: {state.critic_confidence:.2f}")
        print(f"[CRITIC] Retrieval score:   {retrieval_score:.4f}")
        print(f"[CRITIC] confidence_final:  {state.confidence_final:.4f}")
        if state.validation_issues:
            print(f"[CRITIC] Issues: {state.validation_issues}")

        return state