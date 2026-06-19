from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from app.middleware.auth import get_current_user
from app.services.retrieval.hybrid_search import HybridSearchEngine

router = APIRouter(prefix="/retrieval", tags=["retrieval"])

class SearchRequest(BaseModel):
    """Search request"""
    query: str
    top_k: int = 5
    include_vector_score: bool = False

class SearchResult(BaseModel):
    """Single search result"""
    doc_id: str
    chunk_index: int
    text: str
    combined_score: float
    vector_score: float = 0
    keyword_score: float = 0
    search_type: str

class SearchResponse(BaseModel):
    """Search response"""
    query: str
    results: list[SearchResult]
    total_results: int
    search_time_ms: float

@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Hybrid search: Vector + Keyword
    
    Returns top-k most relevant chunks
    """
    import time
    start_time = time.time()
    
    try:
        # Validate query
        if not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
        
        if len(request.query) > 500:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query too long (max 500 chars)"
            )
        
        # Perform hybrid search
        search_engine = HybridSearchEngine()
        results = await search_engine.search(
            query=request.query,
            user_id=user_id,
            top_k=request.top_k
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "doc_id": result['doc_id'],
                "chunk_index": result['chunk_index'],
                "text": result['text'][:500],  # Limit text length
                "combined_score": round(result['combined_score'], 3),
                "vector_score": round(result.get('vector_score', 0), 3),
                "keyword_score": round(result.get('keyword_score', 0), 3),
                "search_type": result.get('search_type', 'hybrid')
            })
        
        # Calculate time
        search_time_ms = (time.time() - start_time) * 1000
        
        return {
            "query": request.query,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "search_time_ms": round(search_time_ms, 1)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )