from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from app.middleware.auth import get_current_user
from app.agents.orchestrator import AgentOrchestrator

router = APIRouter(prefix="/agents", tags=["agents"])

class ChatRequest(BaseModel):
    """Chat request"""
    message: str
    top_k: int = 5

class ChatResponse(BaseModel):
    """Chat response"""
    answer: str
    sources: list[dict]
    confidence: float
    search_time_ms: float

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Chat endpoint using agent orchestration
    
    Flows through:
    1. Planner → Decides strategy
    2. Retriever → Searches documents
    3. Tool Agent → External tools
    4. Critic → Validates answer
    5. Answer Agent → Final response
    """
    
    # Validate input
    if not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty"
        )
    
    if len(request.message) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message too long (max 1000 chars)"
        )
    
    try:
        # Run agent orchestrator
        orchestrator = AgentOrchestrator()
        result = await orchestrator.process(
            question=request.message,
            user_id=user_id
        )
        
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            search_time_ms=result["search_time_ms"]
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent error: {str(e)}"
        )