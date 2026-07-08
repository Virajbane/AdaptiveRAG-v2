from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from app.db.mongodb.client import get_db
from app.middleware.auth import get_current_user
from app.utils.validators import InputValidator
from datetime import datetime
from bson import ObjectId
import bcrypt
import asyncio

router = APIRouter(prefix="/auth", tags=["auth"])


# ============================================
# PYDANTIC MODELS
# ============================================

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str


class LoginRequest(BaseModel):
    email: str
    password: str


# ============================================
# REGISTER ENDPOINT
# ============================================

@router.post("/register")
async def register(request: RegisterRequest):
    """Register a new user — open to anyone, no roles."""
    try:
        # 1. VALIDATE INPUTS
        email = InputValidator.validate_email(request.email)
        password = InputValidator.validate_password(request.password)

        name = request.name.strip()
        if not name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Name is required"
            )

        # 2. GET DATABASE
        db = await get_db()

        # 3. CHECK IF USER ALREADY EXISTS
        existing_user = await db["users"].find_one({"email": email})
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered"
            )

        # 4. HASH PASSWORD
        hashed_password = await asyncio.to_thread(
            lambda: bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")
        )

        # 5. CREATE USER DOCUMENT
        user_data = {
            "email": email,
            "name": name,
            "password_hash": hashed_password,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "is_active": True
        }

        # 6. INSERT INTO DATABASE
        result = await db["users"].insert_one(user_data)
        user_id = str(result.inserted_id)

        # 7. GENERATE TOKEN — log user in immediately after registering
        from app.services.auth_service import JWTService
        access_token = JWTService.create_access_token(user_id=user_id)

        return {
            "user_id": user_id,
            "email": email,
            "name": name,
            "access_token": access_token,
            "token_type": "bearer",
            "message": "User registered successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


# ============================================
# LOGIN ENDPOINT
# ============================================

@router.post("/login")
async def login(request: LoginRequest):
    """Login with email and password — open to anyone."""
    try:
        # 1. VALIDATE EMAIL
        email = InputValidator.validate_email(request.email)

        # 2. GET DATABASE
        db = await get_db()

        # 3. FIND USER
        user = await db["users"].find_one({"email": email})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        # 4. VERIFY PASSWORD
        is_valid = await asyncio.to_thread(
            bcrypt.checkpw,
            request.password.encode("utf-8"),
            user["password_hash"].encode("utf-8")
        )
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        # 5. GENERATE TOKEN
        from app.services.auth_service import JWTService
        access_token = JWTService.create_access_token(
            user_id=str(user["_id"])
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": str(user["_id"]),
            "email": user["email"],
            "name": user.get("name", "")
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


# ============================================
# GET CURRENT USER PROFILE
# ============================================

@router.get("/me")
async def get_profile(user_id: str = Depends(get_current_user)):
    """Get current user profile."""
    try:
        db = await get_db()
        user = await db["users"].find_one({"_id": ObjectId(user_id)})

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        return {
            "user_id": str(user["_id"]),
            "email": user["email"],
            "name": user.get("name", ""),
            "created_at": user.get("created_at"),
            "updated_at": user.get("updated_at")
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching user: {str(e)}"
        )