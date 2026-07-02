"""
Usage: cd backend && python -m eval.ragas_eval.run_ragas
Hits the REAL pipeline (same convention as run_eval.py), then scores with Ragas.
"""
import asyncio, json
from pathlib import Path
from eval.harness.pipeline_runner import run_single
from eval.harness.state_capture import to_ragas_row
from eval.ragas_eval.retrieval_metrics import score_retrieval
from eval.ragas_eval.generation_metrics import score_generation

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "generation_golden.jsonl"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _to_pandas(result):
    """
    score_generation() may return either a ragas EvaluationResult
    (has .to_pandas()) or a plain pandas DataFrame, depending on whether
    the installed ragas version supports EvaluationResult.merge() -- see
    the fallback path in generation_metrics.score_generation(). Normalize
    here so callers don't need to care which one they got.
    """
    return result.to_pandas() if hasattr(result, "to_pandas") else result


async def main():
    cases = [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]
    rows = []
    for case in cases:
        result = await run_single(case["question"], user_id=case["user_id"])
        rows.append(to_ragas_row(result, ground_truth_answer=case.get("ground_truth_answer", "")))

    gen_result = score_generation(rows)
    retr_result = score_retrieval(rows)

    RESULTS_DIR.mkdir(exist_ok=True)
    _to_pandas(gen_result).to_json(RESULTS_DIR / "ragas_generation.json", orient="records", indent=2)
    _to_pandas(retr_result).to_json(RESULTS_DIR / "ragas_retrieval.json", orient="records", indent=2)
    print("✅ Ragas results written to eval/results/")

if __name__ == "__main__":
    asyncio.run(main())