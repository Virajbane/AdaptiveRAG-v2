# eval/deepeval_suite/test_regression.py
"""
Every entry in regression_cases.jsonl is a PAST BUG. This file is the
permanent net that stops a fixed bug from silently reappearing after a
prompt tweak, model swap, or refactor.

Add a new case here EVERY TIME you fix a production bug (see runbook §9).
"""
import pytest, json
from pathlib import Path
from eval.harness.pipeline_runner import run_single

CASES = [json.loads(l) for l in
         (Path(__file__).resolve().parents[1] / "datasets" / "regression_cases.jsonl").read_text().splitlines()
         if l.strip()]

@pytest.mark.regression
@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
async def test_known_bug_stays_fixed(case):
    result = await run_single(case["question"], user_id=case["user_id"])

    if "expected_sources" in case:
        assert result.sources_needed == case["expected_sources"], (
            f"REGRESSION: routing for '{case['question']}' reverted to {result.sources_needed}, "
            f"expected {case['expected_sources']} (bug: {case.get('bug_ref', 'unknown')})"
        )

    for forbidden in case.get("must_not_contain", []):
        assert forbidden.lower() not in result.answer.lower(), (
            f"REGRESSION: answer for '{case['question']}' contains forbidden phrase '{forbidden}' "
            f"(bug: {case.get('bug_ref', 'unknown')})"
        )