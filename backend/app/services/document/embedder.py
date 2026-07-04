from ollama import Client
from typing import List
import asyncio

class EmbeddingGenerator:
    """Generate embeddings using Ollama"""

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self.client = Client(host=ollama_url)
        self.model = model

    async def embed_text(self, text: str, task: str = "search_document") -> List[float]:
        """
        Embed text using Ollama
        Returns 768-dimensional vector

        task: "search_document" for chunks being indexed, "search_query"
        for the incoming user question at retrieval time. nomic-embed-text
        is trained with these prefixes to produce an asymmetric embedding
        space for retrieval — without them, document and query vectors
        don't align well and similarity scores come back near-random
        (this was the root cause of every chunk scoring <0.02 regardless
        of relevance, seen in the 2026-07-04 eval).
        """
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self._embed_sync,
                text,
                task
            )
            return embedding
        except Exception as e:
            raise ValueError(f"Error embedding text: {str(e)}")

    def _embed_sync(self, text: str, task: str = "search_document") -> List[float]:
        """Synchronous embedding (for thread pool)"""
        prefixed_text = f"{task}: {text}"
        response = self.client.embeddings(
            model=self.model,
            prompt=prefixed_text
        )
        return response["embedding"]

    async def embed_batch(self, texts: List[str], task: str = "search_document") -> List[List[float]]:
        """Embed multiple texts (more efficient)"""
        embeddings = []
        for text in texts:
            embedding = await self.embed_text(text, task=task)
            embeddings.append(embedding)
        return embeddings