"""Auth API router — register, login, refresh."""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from app.auth.dependencies import get_current_user
from app.auth.schemas import TokenResponse, UserLogin, UserRegister, UserResponse
from app.auth.service import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    decode_token,
    register_user,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse)
async def register(body: UserRegister):
    """Register a new user account."""
    user = await register_user(body.username, body.password)
    if user is None:
        raise HTTPException(status_code=409, detail="Username already taken")
    return UserResponse(**user)


@router.post("/login", response_model=TokenResponse)
async def login(body: UserLogin):
    """Authenticate and receive JWT tokens."""
    user = await authenticate_user(body.username, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    access_token = create_access_token(user["id"], user["username"])
    refresh_token = create_refresh_token(user["id"])

    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: dict):
    """Exchange a refresh token for new access + refresh tokens."""
    refresh_token = body.get("refresh_token", "")
    payload = decode_token(refresh_token)
    if payload is None or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user_id = payload.get("sub", "")
    # We don't store username in refresh token, so get it from payload or lookup
    username = payload.get("username", user_id)

    new_access = create_access_token(user_id, username)
    new_refresh = create_refresh_token(user_id)

    return TokenResponse(access_token=new_access, refresh_token=new_refresh)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get the current authenticated user's info."""
    return UserResponse(
        id=current_user["id"],
        username=current_user["username"],
        created_at="",
    )
