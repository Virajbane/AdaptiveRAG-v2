# eval/deepeval_suite/custom_metrics.py
"""Domain-specific metric using DeepEval's GEval — an LLM-graded rubric
metric for things Ragas/DeepEval's built-ins don't cover, e.g. 'did the
answer correctly cite which document it came from'."""
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

citation_accuracy_metric = GEval(
    name="CitationAccuracy",
    criteria=(
        "Determine whether the actual output correctly attributes claims to the "
        "source document(s) provided in retrieval_context, without inventing a "
        "source or citing a document that wasn't retrieved."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.7,
)