from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from app.middleware.auth import get_current_user
from app.services.tools.registry import tool_registry

router = APIRouter(prefix="/tools", tags=["tools"])

class ToolExecutionRequest(BaseModel):
    """Request to execute a tool"""
    tool_name: str
    params: dict = {}

class ToolListResponse(BaseModel):
    """List of available tools"""
    tools: list

class CalculateRequest(BaseModel):
    expression: str

class WebSearchRequest(BaseModel):
    """Web search request"""
    query: str
    max_results: int = 5

@router.get("/available")
async def list_tools(user_id: str = Depends(get_current_user)):
    """
    List all available tools
    """
    tools = tool_registry.get_tools()
    return {
        "tools": [
            {
                "name": tool["name"],
                "description": tool["description"],
                "params": tool["params"]
            }
            for tool in tools.values()
        ],
        "count": len(tools)
    }

@router.post("/execute")
async def execute_tool(
    request: ToolExecutionRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Execute a tool
    
    Example:
    {
        "tool_name": "calculator",
        "params": {"expression": "2 + 2"}
    }
    """
    
    try:
        result = await tool_registry.execute_tool(
            request.tool_name,
            **request.params
        )
        
        return {
            "tool": request.tool_name,
            "result": result,
            "success": "error" not in result
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tool execution failed: {str(e)}"
        )

@router.post("/calculate")
async def calculate(
    request: CalculateRequest,
    user_id: str = Depends(get_current_user)
):
    from app.services.tools.calculator import calculator_tool
    try:
        result = await calculator_tool.calculate(request.expression)
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Calculation failed: {str(e)}")


@router.post("/web-search")
async def web_search(
    request: WebSearchRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Quick web search endpoint
    """
    
    from app.services.tools.web_search import web_search_tool
    
    if not web_search_tool:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Web search not available"
        )
    
    try:
        result = await web_search_tool.search(
            request.query,
            max_results=request.max_results
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )