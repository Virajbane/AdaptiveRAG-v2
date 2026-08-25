from typing import Dict, Callable
from app.services.tools.calculator import calculator_tool
from app.services.tools.sql_executor import sql_executor_tool
from app.services.tools.weather import weather_tool
from app.services.tools.slack_tool import slack_tool
from app.services.tools.email_tool import email_tool
import app.services.tools.web_search as web_search_module  # import the MODULE, not the variable

class ToolRegistry:
    """
    Central registry of available tools.
    
    2026-08-09 FIX: Imports weather, slack_post, and send_email tools
    from separate tool files for better organization and consistency.
    """

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
            "weather": {
                "name": "weather",
                "description": "Get current weather for a location using OpenWeatherMap API",
                "callable": weather_tool.get_weather,
                "params": {
                    "location": "City name (e.g., 'London', 'Mumbai')"
                }
            },
            "slack_post": {
                "name": "slack_post",
                "description": "Post a message to a Slack channel",
                "callable": slack_tool.post_message,
                "params": {
                    "channel": "Slack channel (e.g., '#new-channel')",
                    "message": "Message to post"
                }
            },
            # 2026-08-22 STAGE 16+ FIX (root cause 6.5): slack_search is the
            # concrete tool for Slack message-search routing. Requires a user
            # token (xoxp-...) with search:read scope in SLACK_BOT_TOKEN.
            "slack_search": {
                "name": "slack_search",
                "description": "Search Slack message history for a keyword or topic",
                "callable": slack_tool.search_messages,
                "params": {
                    "query": "Keyword or phrase to search for",
                    "channel": "(optional) Slack channel to restrict search to",
                    "user": "(optional) Slack user ID to filter by sender",
                    "count": "(optional) Max number of results (default 10)"
                }
            },
            "send_email": {
                "name": "send_email",
                "description": "Send an email via Gmail SMTP",
                "callable": email_tool.send_email,
                "params": {
                    "subject": "Email subject",
                    "body": "Email body",
                    "to_email": "Recipient email address"
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
        """
        Execute a tool with given parameters.
        
        2026-08-09 FIX: Now properly handles all 6 tools including
        weather, slack_post, and send_email.
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return {"error": f"Tool '{tool_name}' not found"}

        try:
            result = await tool["callable"](**kwargs)
            return result
        except Exception as e:
            import traceback
            print(f"[REGISTRY] Tool '{tool_name}' execution error: {e}")
            traceback.print_exc()
            return {"error": f"Tool execution failed: {str(e)}"}

# Global registry instance
tool_registry = ToolRegistry()