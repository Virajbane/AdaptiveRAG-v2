from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from app.middleware.auth import get_current_user
import app.services.memory.manager as mm_module
from app.db.models.memory_models import SessionMemoryRequest

router = APIRouter(prefix="/memory", tags=["memory"])

class MemoryRequest(BaseModel):
    """Request to get memory"""
    session_id: str

@router.post("/load")
async def load_memory(
    request: MemoryRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Load conversation memory for a session
    
    Returns conversation history and summaries
    """
    
    if not mm_module.memory_manager:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Memory system not initialized"
        )
    
    try:
        context = await mm_module.memory_manager.load_context(user_id, request.session_id)
        return {
            "session_id": request.session_id,
            "history": context["history"],
            "summaries": context["summaries"],
            "preferences": context["preferences"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading memory: {str(e)}"
        )

@router.post("/save-summary")
async def save_summary(
    request: SessionMemoryRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Save session summary to long-term memory
    """
    
    if not mm_module.memory_manager:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Memory system not initialized"
        )
    
    try:
        success = await mm_module.memory_manager.long_term.save_session_summary(
            user_id=user_id,
            session_id=request.session_id,
            summary=request.summary,
            topics=request.topics
        )
        
        if success:
            return {"message": "Summary saved", "session_id": request.session_id}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save summary"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving summary: {str(e)}"
        )

@router.get("/history/{session_id}")
async def get_history(
    session_id: str,
    user_id: str = Depends(get_current_user)
):
    """
    Get conversation history for a session
    """
    
    if not mm_module.memory_manager:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Memory system not initialized"
        )
    
    try:
        history = await mm_module.memory_manager.short_term.get_history(user_id, session_id)
        return {
            "session_id": session_id,
            "messages": history,
            "count": len(history)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting history: {str(e)}"
        )

@router.delete("/session/{session_id}")
async def clear_session(
    session_id: str,
    user_id: str = Depends(get_current_user)
):
    """
    Clear session history
    """
    
    if not mm_module.memory_manager:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Memory system not initialized"
        )
    
    try:
        success = await mm_module.memory_manager.short_term.clear_session(user_id, session_id)
        return {
            "message": "Session cleared",
            "session_id": session_id,
            "success": success
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing session: {str(e)}"
        )