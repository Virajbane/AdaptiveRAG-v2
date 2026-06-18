from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import router




app = FastAPI(
    title="RAG 2.0 System API",
    description="Enterprise-grade Adaptive RAG System",
    version="1.0.0"
)
app.include_router(router)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "api": "up"
        }
    }

@app.get("/")
async def root():
    return {
        "message": "RAG 2.0 System API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)