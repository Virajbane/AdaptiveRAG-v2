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
    LLM provider wrapper - handles Ollama (local dev) / Groq (production)

    2026-07-08: added Groq as a hosted-inference path for production
    deployment (Render free tier has no GPU and not enough RAM for
    qwen2.5:7b). Selection is automatic: if settings.GROQ_API_KEY is
    set, Groq is used; otherwise falls back to local Ollama. This means
    local dev (.env with no GROQ_API_KEY) is untouched, and production
    (Render env vars with GROQ_API_KEY set) automatically switches over
    with no code change needed per environment.
    """

    def __init__(self, model: str = None, num_ctx: int = DEFAULT_NUM_CTX):
        self.use_groq = bool(settings.GROQ_API_KEY)
        self.num_ctx = num_ctx

        if self.use_groq:
            self.model = model or settings.GROQ_MODEL
        else:
            self.model = model or settings.OLLAMA_MODEL
            self.base_url = settings.OLLAMA_BASE_URL

    async def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """
        Generate text using Groq (production) or Ollama (local dev)

        Raises:
            Exception: propagates the real underlying error (e.g. model
            not found, connection refused, timeout) so callers can
            distinguish "LLM call failed" from "LLM returned bad JSON".
            Do NOT swallow exceptions here and return them as strings -
            that disguises real failures as model output and breaks
            downstream JSON parsing with a misleading error.
        """
        if self.use_groq:
            return await self._generate_groq(prompt, max_tokens)
        return await self._generate_ollama(prompt, max_tokens)

    async def _generate_groq(self, prompt: str, max_tokens: int) -> str:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    async def _generate_ollama(self, prompt: str, max_tokens: int) -> str:
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

    async def acomplete(
        self,
        system: str = "",
        prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> object:
        """
        Compatibility wrapper matching the planner/agents' interface.

        2026-08-09 FIX: The planner and other agents call with 
        (system, prompt, temperature, max_tokens) signature, but 
        our generate() only takes (prompt, max_tokens). This adapter 
        bridges the gap by combining system+prompt and delegating to 
        generate(), then wrapping the result in an object with a .text 
        attribute that callers expect.

        Args:
            system: System/instruction prompt (context/rules for the LLM)
            prompt: User/question prompt (the actual query)
            temperature: Sampling temperature (currently ignored - Ollama/Groq 
                        may not support in async API, but accepted for compatibility)
            max_tokens: Maximum output tokens

        Returns:
            Object with .text attribute containing the response string

        Example:
            response = await llm.acomplete(
                system="Classify the question...",
                prompt="What is the weather?",
                temperature=0,
                max_tokens=40,
            )
            result = response.text  # Get the response text
        """
        # Combine system prompt (instructions) with user prompt (question)
        # System gives the LLM context/rules, prompt is what we actually ask
        full_prompt = f"{system}\n\n{prompt}".strip() if system else prompt

        # Call the main generate method (it handles Groq vs Ollama internally)
        response_text = await self.generate(full_prompt, max_tokens)

        # Wrap the text in an object with .text attribute
        # This matches what callers (planner, agents) expect
        class TextResponse:
            """Simple response wrapper with .text attribute"""
            def __init__(self, text: str):
                self.text = text

        return TextResponse(response_text)