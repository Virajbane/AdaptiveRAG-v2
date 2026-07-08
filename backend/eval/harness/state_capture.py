# eval/harness/state_capture.py
"""Normalizes a PipelineResult (or raw AgentState) into the flat dict shape
Ragas 0.4.x's SingleTurnSample expects: user_input, response, retrieved_contexts, reference."""

def to_ragas_row(pipeline_result, ground_truth_answer: str = "", ground_truth_contexts: list[str] | None = None) -> dict:
    return {
        "user_input": pipeline_result.question,
        "response": pipeline_result.answer,
        "retrieved_contexts": pipeline_result.contexts or [""],   # Ragas errors on empty list
        "reference": ground_truth_answer,
        "reference_contexts": ground_truth_contexts or [],
    }