# eval/deepeval_suite/conftest.py
import pytest
from deepeval.models import GeminiModel
from eval.config import JUDGE_MODEL


@pytest.fixture(scope="session")
def judge_model():
    """Shared Gemini judge model for all DeepEval metrics in this suite.
    Uses Google's free-tier API — reads GOOGLE_API_KEY from env automatically.
    """
    return GeminiModel(
        model=JUDGE_MODEL,
        temperature=0,
    )