# eval/deepeval_suite/test_hallucination.py
import pytest, json
from pathlib import Path
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from eval.harness.pipeline_runner import run_single
from eval.config import THRESHOLDS

CASES = [json.loads(l) for l in
         (Path(__file__).resolve().parents[1] / "datasets" / "generation_golden.jsonl").read_text().splitlines()
         if l.strip()]

@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
async def test_no_hallucination(case):
    result = await run_single(case["question"], user_id=case["user_id"])

    test_case = LLMTestCase(
        input=case["question"],
        actual_output=result.answer,
        context=result.contexts,          # HallucinationMetric checks against provided context
        retrieval_context=result.contexts,
    )

    hallucination = HallucinationMetric(threshold=THRESHOLDS["hallucination_max"])
    relevancy = AnswerRelevancyMetric(threshold=THRESHOLDS["answer_relevancy"])

    assert_test(test_case, [hallucination, relevancy])