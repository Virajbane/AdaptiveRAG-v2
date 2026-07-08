"""
Run the entire evaluation suite in one command, cheapest checks first.

WHY THIS ORDER:
    Router eval costs ~seconds (no retrieval, no generation).
    Retrieval eval costs ~seconds (no generation).
    Regression suite costs a bit more (full pipeline, but few cases).
    Ragas + DeepEval full suite costs the most (LLM-judge on every case).

    If router eval fails, something is fundamentally broken (wrong model,
    Ollama down, bad imports) — no point burning minutes on Ragas only to
    hit the same root cause. This script stops at the first failing stage
    so you fix issues in the cheapest possible order.

HOW TO RUN:
    cd backend
    python -m eval.run_all

    Add --continue-on-fail to run every stage regardless (useful once
    everything is stable and you just want the full report at the end):
    python -m eval.run_all --continue-on-fail
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]

STAGES = [
    ("Router eval",        [sys.executable, "-m", "eval.router_eval.run_router_eval"]),
    ("Retrieval eval",     [sys.executable, "-m", "eval.retrieval_eval.run_retrieval_eval"]),
    ("Regression suite",   [sys.executable, "-m", "pytest", "eval/deepeval_suite/test_regression.py", "-m", "regression", "-v"]),
    ("Ragas full suite",   [sys.executable, "-m", "eval.ragas_eval.run_ragas"]),
    ("DeepEval full suite",[sys.executable, "-m", "pytest", "eval/deepeval_suite/", "-v", "--html=eval/reports/full_deepeval.html", "--self-contained-html"]),
    ("Report builder",     [sys.executable, "-m", "eval.reports.report_builder"]),
]


def run_stage(name: str, cmd: list[str]) -> bool:
    print("\n" + "=" * 60)
    print(f"STAGE: {name}")
    print("=" * 60)
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=BACKEND_DIR)
    elapsed = time.perf_counter() - start
    passed = result.returncode == 0
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"\n{status}  ({elapsed:.1f}s)  [{name}]")
    return passed


def main():
    continue_on_fail = "--continue-on-fail" in sys.argv

    summary = []
    for name, cmd in STAGES:
        passed = run_stage(name, cmd)
        summary.append((name, passed))
        if not passed and not continue_on_fail:
            print(f"\n🛑 Stopping — '{name}' failed. Fix this before running later "
                  f"(more expensive) stages. Use --continue-on-fail to override.")
            break

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in summary:
        print(f"  {'✅' if passed else '❌'}  {name}")

    if all(p for _, p in summary) and len(summary) == len(STAGES):
        print("\n✅ All stages passed. See eval/reports/summary.md and eval/reports/full_deepeval.html")
    else:
        print("\n⚠️  Not all stages ran or passed — see above.")


if __name__ == "__main__":
    main()