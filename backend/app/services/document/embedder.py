from ollama import Client
from typing import List
import asyncio

class EmbeddingGenerator:
    """Generate embeddings using Ollama"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self.client = Client(host=ollama_url)
        self.model = model
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Embed text using Ollama
        Returns 768-dimensional vector
        """
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self._embed_sync,
                text
            )
            return embedding
        except Exception as e:
            raise ValueError(f"Error embedding text: {str(e)}")
    
    def _embed_sync(self, text: str) -> List[float]:
        """Synchronous embedding (for thread pool)"""
        response = self.client.embeddings(
            model=self.model,
            prompt=text
        )
        return response["embedding"]
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts (more efficient)"""
        embeddings = []
        for text in texts:
            embedding = await self.embed_text(text)
            embeddings.append(embedding)
        return embeddings