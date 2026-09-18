"""Pydantic v2 schemas for auth endpoints.

These are the data contracts for signup, login, and user info responses.
Every cross-component payload is a typed Pydantic model — no raw dicts.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator

# bcrypt hashes at most the first 72 bytes and silently ignores the rest, so a
# 200-character passphrase is no stronger than its first 72 bytes. Reject anything
# longer outright rather than accepting a password we only partly check.
MAX_PASSWORD_BYTES = 72
MIN_PASSWORD_LENGTH = 8


# ── Requests ────────────────────────────────────────────────────


class SignupRequest(BaseModel):
    """POST /auth/signup body."""

    email: EmailStr
    password: str = Field(..., min_length=MIN_PASSWORD_LENGTH)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        """Lower-case the address so one person cannot register twice.

        The uniqueness check is a plain equality match on this column, so
        "Demo@Example.com" and "demo@example.com" would otherwise become two
        separate accounts and login would depend on how the user typed it.
        """
        return v.strip().lower()

    @field_validator("password")
    @classmethod
    def check_password_strength(cls, v: str) -> str:
        if len(v.encode("utf-8")) > MAX_PASSWORD_BYTES:
            raise ValueError(
                f"Password must be at most {MAX_PASSWORD_BYTES} bytes long"
            )
        if not any(c.isalpha() for c in v):
            raise ValueError("Password must contain at least one letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one number")
        return v


class LoginRequest(BaseModel):
    """POST /auth/login body."""

    email: EmailStr
    password: str

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.strip().lower()


# ── Responses ───────────────────────────────────────────────────


class TokenResponse(BaseModel):
    """Returned after successful signup or login."""

    access_token: str
    token_type: str = "bearer"
    # Lets the client know when to stop trusting this token instead of finding out
    # from the first 401 mid-action.
    expires_in: int = 0


class UserResponse(BaseModel):
    """Public user representation — never exposes hashed_password."""

    id: uuid.UUID
    email: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
