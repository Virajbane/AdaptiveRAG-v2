from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["v1"])

# Health endpoint
@router.get("/health")
async def health():
    return {"status": "ok"}

# Later: Add auth, chat, documents endpoints here