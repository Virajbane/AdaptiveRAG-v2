# eval/ragas_eval/retrieval_metrics.py
from ragas import evaluate, EvaluationDataset
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall, ContextUtilization
from eval.harness.llm_judge_provider import get_ragas_llm, get_ragas_embeddings

# NOTE: "Context Relevance" as a standalone metric isn't part of ragas==0.2.10's
# API (it existed under a different name in older Ragas versions and was
# restructured). ContextUtilization is the closest available equivalent —
# it checks whether the MOST relevant retrieved chunks were ranked near the
# top, which overlaps with what "relevance" was meant to capture, but isn't
# a 1:1 replacement. Context Precision/Recall below are the two metrics that
# matter most for retrieval quality either way.
RETRIEVAL_METRICS = [
    LLMContextPrecisionWithReference(),  # did we rank relevant chunks higher?
    LLMContextRecall(),                  # did we retrieve everything needed?
    ContextUtilization(),                # were the top-ranked chunks actually the useful ones?
]

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
    )