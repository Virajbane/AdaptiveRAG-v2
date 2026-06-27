import json
from app.agents.base import BaseAgent
from app.agents.state import AgentState
from app.services.tools.registry import tool_registry

class ToolAgent(BaseAgent):
    """
    Tool Agent: Executes external tools
    
    Available tools:
    - web_search: Search the web
    - calculator: Math calculations
    - sql_query: Database queries
    """
    
    async def _execute(self, state: AgentState) -> AgentState:
        """Execute tools if needed"""
        
        # Check if tools are needed
        if "web" not in state.sources_needed and "tools" not in state.sources_needed:
            return state
        
        try:
    # For now, use web search if web or tools is needed
            if "web" in state.sources_needed or "tools" in state.sources_needed:
                print("[TOOL] Executing web search...")
                
                result = await tool_registry.execute_tool(
                    "web_search",
                    query=state.question,
                    max_results=5
                )
                
                state.tool_results["web_search"] = result
                print(f"[TOOL] Web search complete: {result.get('count', 0)} results")

        except Exception as e:
            print(f"[TOOL] Tool execution error: {e}")
            state.error = f"Tool agent error: {str(e)}"

        return state