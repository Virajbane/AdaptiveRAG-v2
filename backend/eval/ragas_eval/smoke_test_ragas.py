# eval/ragas_eval/smoke_test_ragas.py
"""
30-second sanity check: does one row survive to_ragas_row -> score_generation/score_retrieval
without silently producing NaN or raising a pydantic validation error?

Usage: cd backend && python -m eval.ragas_eval.smoke_test_ragas
"""
import asyncio
from eval.harness.pipeline_runner import run_single
from eval.harness.state_capture import to_ragas_row
from eval.ragas_eval.retrieval_metrics import score_retrieval
from eval.ragas_eval.generation_metrics import score_generation

# Pick one real, known-answerable question from your system.
SMOKE_QUESTION = "What is our refund policy for enterprise customers?"
SMOKE_USER_ID = "eval_user_1"
SMOKE_GROUND_TRUTH = "Enterprise customers can request a full refund within 30 days of purchase."


async def main():
    result = await run_single(SMOKE_QUESTION, user_id=SMOKE_USER_ID)
    row = to_ragas_row(result, ground_truth_answer=SMOKE_GROUND_TRUTH)

    print("\n--- Row shape sent to ragas ---")
    for k, v in row.items():
        preview = (str(v)[:80] + "...") if len(str(v)) > 80 else v
        print(f"  {k}: {preview}")

    print("\n--- Running score_generation ---")
    gen_result = score_generation([row])
    gen_df = gen_result.to_pandas()
    print(gen_df.to_string())

    print("\n--- Running score_retrieval ---")
    retr_result = score_retrieval([row])
    retr_df = retr_result.to_pandas()
    print(retr_df.to_string())

    # Explicit check: did any metric column come back all-NaN?
    nan_cols = [c for c in gen_df.columns if gen_df[c].isna().all()] + \
               [c for c in retr_df.columns if retr_df[c].isna().all()]
    if nan_cols:
        print(f"\n⚠️  WARNING: these columns are all-NaN, field mapping is likely still broken: {nan_cols}")
    else:
        print("\n✅ All metric columns produced values — schema is compatible.")


if __name__ == "__main__":
    asyncio.run(main())