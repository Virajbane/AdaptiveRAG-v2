# eval/config.py
import os

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gemini-2.5-flash-lite")
EMBEDDING_MODEL = os.getenv("EVAL_EMBEDDING_MODEL", "nomic-embed-text:latest")

THRESHOLDS = {
    "faithfulness": 0.75,
    "answer_relevancy": 0.70,
    "context_precision": 0.65,
    "context_recall": 0.70,
    "hallucination_max": 0.25,
}