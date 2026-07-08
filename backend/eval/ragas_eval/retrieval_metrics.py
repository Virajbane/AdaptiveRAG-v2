# eval/ragas_eval/retrieval_metrics.py
from ragas import evaluate, EvaluationDataset
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall, ContextRelevance
from ragas.run_config import RunConfig
from eval.harness.llm_judge_provider import get_ragas_llm, get_ragas_embeddings

RETRIEVAL_METRICS = [
    LLMContextPrecisionWithReference(),  # did we rank relevant chunks higher?
    LLMContextRecall(),                  # did we retrieve everything needed?
    ContextRelevance(),                  # how relevant is retrieved context to the question?
]

# Local Ollama serves one model at a time and effectively serializes calls on
# most dev boxes. Ragas defaults to max_workers=16, which fires many concurrent
# judge/embedding calls at once -- against local Ollama this causes model-swap
# thrashing (chat model <-> embedding model) and requests queueing past the
# 180s default timeout. Cap concurrency and give each call more room.
LOCAL_OLLAMA_RUN_CONFIG = RunConfig(timeout=300, max_workers=2, max_retries=3)

def score_retrieval(rows: list[dict]):
    """
    rows: [{question, contexts, ground_truth, reference_contexts}, ...]
    Returns a ragas EvaluationResult (convertible to pandas via .to_pandas()).
    """
    dataset = EvaluationDataset.from_list(rows)
    return evaluate(
        dataset=dataset,
        metrics=RETRIEVAL_METRICS,
        llm=get_ragas_llm(),
        embeddings=get_ragas_embeddings(),
        run_config=LOCAL_OLLAMA_RUN_CONFIG,
        raise_exceptions=True,  # TEMP: surface real errors instead of swallowed TimeoutError
    )