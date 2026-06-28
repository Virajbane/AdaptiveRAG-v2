from typing import Optional
from app.config.settings import settings


class LLMProvider:
    """
    LLM provider wrapper - handles Ollama/OpenAI/Gemini
    """

    def __init__(self, model: str = None):
        self.model = model or settings.OLLAMA_MODEL
        self.base_url = settings.OLLAMA_BASE_URL

    async def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """
        Generate text using Ollama (local)

        Args:
            prompt: Input prompt
            max_tokens: Max tokens to generate

        Returns:
            Generated text

        Raises:
            Exception: propagates the real underlying error (e.g. model
            not found, connection refused, timeout) so callers can
            distinguish "LLM call failed" from "LLM returned bad JSON".
            Do NOT swallow exceptions here and return them as strings -
            that disguises real failures as model output and breaks
            downstream JSON parsing with a misleading error.
        """
        from ollama import AsyncClient

        client = AsyncClient(host=self.base_url)
        response = await client.generate(
            model=self.model,
            prompt=prompt,
            stream=False,
            options={"num_predict": max_tokens}
        )

        # ollama python client has changed response shape across versions
        # (plain dict vs. typed GenerateResponse) - handle both.
        if isinstance(response, dict):
            return response.get('response', '')
        return getattr(response, 'response', '')