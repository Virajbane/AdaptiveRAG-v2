from typing import Dict, Callable
from app.services.tools.calculator import calculator_tool
from app.services.tools.sql_executor import sql_executor_tool
import app.services.tools.web_search as web_search_module  # import the MODULE, not the variable

class ToolRegistry:
    """Central registry of available tools"""

    def __init__(self):
        self.tools: Dict[str, Dict] = {
            "calculator": {
                "name": "calculator",
                "description": "Perform mathematical calculations",
                "callable": calculator_tool.calculate,
                "params": {
                    "expression": "Math expression (e.g., '2 + 2 * 3')"
                }
            },
            "sql_query": {
                "name": "sql_query",
                "description": "Query database",
                "callable": sql_executor_tool.execute_query,
                "params": {
                    "collection": "Collection name",
                    "query_type": "find|count|aggregate",
                    "query": "Query parameters"
                }
            },
        }

    def get_tools(self) -> Dict:
        """
        Get all available tools.

        Checks web_search_module.web_search_tool live, on every call,
        instead of relying on a value captured once at __init__ time.
        web_search_tool starts as None at import time and only gets
        set inside init_web_search(), which runs later during the
        FastAPI startup event - by the time a request actually comes
        in, the module-level variable has the real value, but only if
        we look it up through the module each time instead of having
        imported the bare name once up front.
        """
        tools = dict(self.tools)  # copy so we never mutate self.tools permanently

        if web_search_module.web_search_tool:
            tools["web_search"] = {
                "name": "web_search",
                "description": "Search the web for information",
                "callable": web_search_module.web_search_tool.search,
                "params": {
                    "query": "Search query",
                    "max_results": "Number of results (default 5)"
                }
            }

        return tools

    def get_tool(self, tool_name: str):
        """Get specific tool"""
        return self.get_tools().get(tool_name)

    async def execute_tool(self, tool_name: str, **kwargs) -> Dict:
        """Execute a tool with given parameters"""
        tool = self.get_tool(tool_name)
        if not tool:
            return {"error": f"Tool '{tool_name}' not found"}

        try:
            result = await tool["callable"](**kwargs)
            return result
        except Exception as e:
            return {"error": str(e)}

# Global registry instance
tool_registry = ToolRegistry()