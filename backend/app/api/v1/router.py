from fastapi import APIRouter
from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.auth import user_router
from app.api.v1.endpoints.documents import router as documents_router

router = APIRouter(prefix="/api/v1", tags=["v1"])

@router.get("/health")
async def health():
    return {"status": "ok"}

router.include_router(auth_router)
router.include_router(user_router)
router.include_router(documents_router)