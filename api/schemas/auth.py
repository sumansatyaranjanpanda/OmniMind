"""Pydantic v2 schemas for auth endpoints.

These are the data contracts for signup, login, and user info responses.
Every cross-component payload is a typed Pydantic model — no raw dicts.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr

# ── Requests ────────────────────────────────────────────────────


class SignupRequest(BaseModel):
    """POST /auth/signup body."""

    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    """POST /auth/login body."""

    email: EmailStr
    password: str


# ── Responses ───────────────────────────────────────────────────


class TokenResponse(BaseModel):
    """Returned after successful signup or login."""

    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    """Public user representation — never exposes hashed_password."""

    id: uuid.UUID
    email: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
