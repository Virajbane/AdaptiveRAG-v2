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
        """
        from ollama import Client
        
        try:
            client = Client(host=self.base_url)
            response = client.generate(
                model=self.model,
                prompt=prompt,
                stream=False
            )
            return response.get('response', '')
        except Exception as e:
            # Fallback: return error message
            return f"Error: {str(e)}"