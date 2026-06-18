from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from app.db.mongodb.models import UserCreate, UserResponse
from app.db.mongodb.queries import UserQueries
from app.services.password_service import PasswordService
from app.services.auth_service import JWTService
from app.config.settings import settings
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.middleware.auth import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])

async def get_db() -> AsyncIOMotorDatabase:
    from app.db.mongodb.client import db
    return db

class TokenResponse(BaseModel):
    user_id: str
    email: str
    name: str
    access_token: str
    token_type: str
    expires_in: int

class LoginRequest(BaseModel):
    email: str
    password: str

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: AsyncIOMotorDatabase = Depends(get_db)):
    # Validate password strength
    is_strong, message = PasswordService.is_strong_password(user_data.password)
    if not is_strong:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)

    # Check if user already exists
    user_queries = UserQueries(db)
    if await user_queries.user_exists(user_data.email):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    # Hash password and create user
    password_hash = PasswordService.hash_password(user_data.password)
    user_id = await user_queries.create_user(
        email=user_data.email,
        name=user_data.name,
        password_hash=password_hash
    )

    user = await user_queries.get_user_by_id(user_id)

    return {
        "user_id": str(user["_id"]),
        "email": user["email"],
        "name": user["name"],
        "created_at": user["created_at"]
    }

@router.post("/login", response_model=TokenResponse)
async def login(credentials: LoginRequest, db: AsyncIOMotorDatabase = Depends(get_db)):
    user_queries = UserQueries(db)

    # Get user by email
    user = await user_queries.get_user_by_email(credentials.email)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    # Verify password
    if not PasswordService.verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    # Check active
    if not user.get("is_active", True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account deactivated")

    # Create token
    access_token = JWTService.create_access_token(
        user_id=str(user["_id"]),
        expires_in_hours=24
    )

    return {
        "user_id": str(user["_id"]),
        "email": user["email"],
        "name": user["name"],
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 86400
    }

@router.get("/users/me")
async def get_me(db: AsyncIOMotorDatabase = Depends(get_db)):
    return {"message": "protected route - coming soon"}

user_router = APIRouter(prefix="/users", tags=["users"])

@user_router.get("/me")
async def get_me(
    user_id: str = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    user_queries = UserQueries(db)
    user = await user_queries.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return {
        "user_id": str(user["_id"]),
        "email": user["email"],
        "name": user["name"],
        "created_at": user["created_at"]
    }