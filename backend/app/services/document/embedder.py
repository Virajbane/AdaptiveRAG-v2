from typing import List
import asyncio
from app.config.settings import settings


class EmbeddingGenerator:
    """
    Generate embeddings - uses Voyage AI (production) or Ollama (local dev).

    2026-07-08: added Voyage AI as the hosted embedding path for production.
    Groq (used for LLM generation) does not offer an embeddings API, so
    embeddings were still hardcoded to a local Ollama instance that does
    not exist on Render, silently timing out on every document upload
    and every search query. Selection mirrors LLMProvider's pattern: if
    settings.VOYAGE_API_KEY is set, Voyage is used; otherwise falls back
    to local Ollama for dev. output_dimension=768 keeps vectors compatible
    with the existing Qdrant collection (created for nomic-embed-text's
    native 768-dim output) with no migration needed.
    """

    def __init__(self, ollama_url: str = None, model: str = None):
        self.use_voyage = bool(settings.VOYAGE_API_KEY)

        if self.use_voyage:
            self.model = model or "voyage-3-lite"
        else:
            self.ollama_url = ollama_url or settings.OLLAMA_BASE_URL
            self.model = model or "nomic-embed-text"
            from ollama import Client
            self.client = Client(host=self.ollama_url)

    async def embed_text(self, text: str, task: str = "search_document") -> List[float]:
        """
        Embed text. Returns a 768-dimensional vector.

        task: "search_document" for chunks being indexed, "search_query"
        for the incoming user question at retrieval time.
        """
        try:
            loop = asyncio.get_event_loop()
            if self.use_voyage:
                embedding = await loop.run_in_executor(
                    None, self._embed_sync_voyage, text, task
                )
            else:
                embedding = await loop.run_in_executor(
                    None, self._embed_sync_ollama, text, task
                )
            return embedding
        except Exception as e:
            raise ValueError(f"Error embedding text: {str(e)}")

    def _embed_sync_voyage(self, text: str, task: str) -> List[float]:
        import voyageai

        vo = voyageai.Client(api_key=settings.VOYAGE_API_KEY)
        # Voyage's input_type distinguishes documents from queries,
        # same asymmetric-embedding concept as nomic-embed-text's prefixes.
        input_type = "query" if task == "search_query" else "document"
        result = vo.embed(
            [text],
            model=self.model,
            input_type=input_type,
            output_dimension=768,
        )
        return result.embeddings[0]

    def _embed_sync_ollama(self, text: str, task: str) -> List[float]:
        prefixed_text = f"{task}: {text}"
        response = self.client.embeddings(
            model=self.model,
            prompt=prefixed_text
        )
        return response["embedding"]

    async def embed_batch(
        self,
        texts: List[str],
        task: str = "search_document",
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> dict:
        """
        Embed multiple texts. A single failing text (after retries) does
        NOT fail the whole batch -- it's recorded and skipped so the
        rest of the document still gets embedded and stored.

        Returns:
            {
                "embeddings": [List[float] | None, ...],  # same length/order as `texts`;
                                                            # None at any index that failed
                "failed_indices": [int, ...],
            }
        """
        embeddings: List = [None] * len(texts)
        failed_indices = []

        for i, text in enumerate(texts):
            success = False
            for attempt in range(1, max_retries + 1):
                try:
                    embeddings[i] = await self.embed_text(text, task=task)
                    success = True
                    break
                except Exception as e:
                    if attempt < max_retries:
                        delay = base_delay * (2 ** (attempt - 1))  # 1s, 2s, 4s
                        print(
                            f"[EMBED RETRY] chunk_index={i} attempt={attempt}/{max_retries} "
                            f"failed ({type(e).__name__}: {e}) -- retrying in {delay}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        print(
                            f"[EMBED FAILED] chunk_index={i} gave up after "
                            f"{max_retries} attempts: {type(e).__name__}: {e}"
                        )

            if not success:
                failed_indices.append(i)

        return {
            "embeddings": embeddings,
            "failed_indices": failed_indices,
        }