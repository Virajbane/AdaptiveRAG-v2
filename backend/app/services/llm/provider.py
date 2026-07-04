from typing import Optional
from app.config.settings import settings

# 2026-07-04 fix: Ollama was crashing mid-generation with
# "wsarecv: An existing connection was forcibly closed by the remote
# host" (500 Internal Server Error) specifically on prompts that
# combined multiple document chunks + multiple web results in the
# same context (AnswerAgent's heaviest case). `ollama ps` at the time
# showed qwen2.5:0.5b AND nomic-embed-text both loaded at 100% GPU
# simultaneously on a 2GB VRAM card - classic VRAM contention, not a
# logic bug. The crash happened on both the original attempt and the
# retry, ruling out a one-off fluke.
#
# num_ctx defaults to whatever the model's Modelfile specifies (4096
# for qwen2.5:0.5b per `ollama ps` CONTEXT column) if not passed
# explicitly. Capping it here bounds the KV-cache VRAM footprint per
# generation call regardless of how large the prompt gets, trading
# some max-context headroom for stability on constrained hardware.
#
# This is a class-level default (not hardcoded per-call) so every
# call site (AnswerAgent, PlannerAgent, CriticAgent, RewriterAgent)
# benefits without each needing to pass options itself. Call sites
# that genuinely need a larger window can still override via the new
# num_ctx param.
#
# Companion fix (not in this file): before starting `ollama serve`,
# set the environment variable OLLAMA_MAX_LOADED_MODELS=1 so Ollama
# unloads one model before loading another instead of holding both
# qwen2.5:0.5b and nomic-embed-text in VRAM at once. On Windows
# PowerShell:
#     $env:OLLAMA_MAX_LOADED_MODELS = "1"
#     ollama serve
DEFAULT_NUM_CTX = 2048


class LLMProvider:
    """
    LLM provider wrapper - handles Ollama/OpenAI/Gemini
    """

    def __init__(self, model: str = None, num_ctx: int = DEFAULT_NUM_CTX):
        self.model = model or settings.OLLAMA_MODEL
        self.base_url = settings.OLLAMA_BASE_URL
        self.num_ctx = num_ctx

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
            options={
                "num_predict": max_tokens,
                # See DEFAULT_NUM_CTX note above - bounds VRAM used by
                # the context window itself, independent of max_tokens
                # (which only bounds the *output* length).
                "num_ctx": self.num_ctx,
                # 2026-07-04: `ollama ps` + nvidia-smi confirmed the GPU
                # is an NVIDIA GeForce GT 710 (2GB VRAM, legacy/entry
                # card). A direct, isolated test call to Ollama (no app
                # involved) reproduced the exact same 500/wsarecv crash,
                # with VRAM climbing to 1857/2048 MiB (191 MiB free) right
                # before it died - a genuine OOM on this card, not a
                # driver timeout or app-level concurrency issue. Given
                # this hardware has a strong CPU (i9) and 32GB system RAM
                # vs. only 2GB VRAM, forcing CPU-only inference trades
                # some speed for actually finishing generation instead of
                # crashing partway through. num_gpu=0 tells Ollama to put
                # zero model layers on GPU, i.e. full CPU inference.
                "num_gpu": 0,
            }
        )

        # ollama python client has changed response shape across versions
        # (plain dict vs. typed GenerateResponse) - handle both.
        if isinstance(response, dict):
            return response.get('response', '')
        return getattr(response, 'response', '')