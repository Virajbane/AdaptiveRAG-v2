from tavily import TavilyClient
from typing import List, Dict

class WebSearchTool:
    """Web search using Tavily API"""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("TAVILY_API_KEY not set in environment")
        self.client = TavilyClient(api_key=api_key)
    
    async def search(
        self,
        query: str,
        max_results: int = 5
    ) -> Dict:
        """
        Search the web using Tavily
        
        Args:
            query: Search query
            max_results: Number of results
        
        Returns:
            {
                "results": [
                    {"title": "...", "url": "...", "snippet": "..."},
                    ...
                ],
                "query": "..."
            }
        """
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                include_answer=True
            )
            
            results = []
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title"),
                    "url": result.get("url"),
                    "snippet": result.get("content"),
                    "source": result.get("source")
                })
            
            return {
                "results": results,
                "query": query,
                "answer": response.get("answer", ""),
                "count": len(results)
            }
        except Exception as e:
            return {
                "results": [],
                "error": str(e),
                "query": query,
                "count": 0
            }

# Global instance
web_search_tool = None

def init_web_search(api_key: str):
    """Initialize web search tool"""
    global web_search_tool
    try:
        web_search_tool = WebSearchTool(api_key)
        print("✅ Web search tool initialized")
    except Exception as e:
        print(f"⚠️  Web search not available: {e}")