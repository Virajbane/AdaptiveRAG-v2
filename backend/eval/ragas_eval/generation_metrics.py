# eval/ragas_eval/generation_metrics.py
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness
from ragas.run_config import RunConfig
from eval.harness.llm_judge_provider import get_ragas_llm, get_ragas_embeddings

# AnswerCorrectness is far more expensive than the other two metrics: it
# does statement generation + classification + similarity scoring, i.e.
# multiple sequential LLM round-trips per row against the same local
# Ollama instance. Running it in the same pool as the cheap metrics is
# what caused a single row to blow past the 300s timeout and, with
# raise_exceptions=True, take down the entire suite (see 2026-07-02
# ragas full suite failure at item 4/15).
#
# Split into two groups with separate RunConfigs so the expensive metric
# gets more time and doesn't compete for Ollama concurrency with the
# cheap ones.
CHEAP_METRICS = [
    Faithfulness(),      # is the answer grounded in the retrieved contexts?
    AnswerRelevancy(),   # does the answer actually address the question?
]
EXPENSIVE_METRICS = [
    AnswerCorrectness(),  # compares against ground_truth (needs labeled data)
]

# See retrieval_metrics.py for why max_workers is capped down from ragas'
# default of 16 -- local Ollama can't handle that much concurrent load.
LOCAL_OLLAMA_RUN_CONFIG = RunConfig(timeout=300, max_workers=2, max_retries=3)

# AnswerCorrectness gets a longer timeout and max_workers=1 so it never
# competes with itself (or anything else) for the single local Ollama
# model instance.
EXPENSIVE_RUN_CONFIG = RunConfig(timeout=600, max_workers=1, max_retries=2)


def score_generation(rows: list[dict]):
    dataset = EvaluationDataset.from_list(rows)
    llm = get_ragas_llm()
    embeddings = get_ragas_embeddings()

    cheap_result = evaluate(
        dataset=dataset,
        metrics=CHEAP_METRICS,
        llm=llm,
        embeddings=embeddings,
        run_config=LOCAL_OLLAMA_RUN_CONFIG,
        # Don't let one stuck/slow row kill the whole suite -- record it
        # as a failed score for that row and keep going. The previous
        # raise_exceptions=True was a debugging TEMP flag; now that the
        # cause (unbounded AnswerCorrectness cost) is known, this is off.
        raise_exceptions=False,
    )

    expensive_result = evaluate(
        dataset=dataset,
        metrics=EXPENSIVE_METRICS,
        llm=llm,
        embeddings=embeddings,
        run_config=EXPENSIVE_RUN_CONFIG,
        raise_exceptions=False,
    )

    # ragas EvaluationResult supports pandas-level combination; merge on
    # the dataframe rather than assuming a `.merge()` helper exists on
    # every ragas version.
    if hasattr(cheap_result, "merge"):
        return cheap_result.merge(expensive_result)

    import pandas as pd

    cheap_df = cheap_result.to_pandas()
    expensive_df = expensive_result.to_pandas()

    # Both dataframes carry the same base columns (user_input, response,
    # retrieved_contexts, reference, ...) since they're scored from the
    # same dataset. Drop ANY column already present in cheap_df from
    # expensive_df before concatenating, so only expensive_df's unique
    # metric columns (e.g. answer_correctness) get added. The previous
    # version excluded the base columns from this drop list, which was
    # backwards -- it left them duplicated and pandas' to_json(orient=
    # "records") refuses to serialize a frame with non-unique columns.
    shared_cols = [c for c in expensive_df.columns if c in cheap_df.columns]
    merged_df = pd.concat(
        [cheap_df, expensive_df.drop(columns=shared_cols, errors="ignore")],
        axis=1,
    )
    return merged_df