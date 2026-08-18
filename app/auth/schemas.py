"""Authentication schemas for user registration and login."""
from typing import Optional
from pydantic import BaseModel, Field


class UserRegister(BaseModel):
    """Request body for user registration."""
    username: str = Field(..., min_length=3, max_length=32, pattern=r"^[a-zA-Z0-9_]+$")
    password: str = Field(..., min_length=6, max_length=128)


class UserLogin(BaseModel):
    """Request body for user login."""
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 1800


class UserResponse(BaseModel):
    """Public user info (no password)."""
    id: str
    username: str
    created_at: str = ""
