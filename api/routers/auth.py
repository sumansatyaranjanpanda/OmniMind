"""Auth router — signup, login, and user-info endpoints.

POST /auth/signup  — create account, return JWT
POST /auth/login   — validate credentials, return JWT
GET  /auth/me      — protected: return current user info
"""

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.deps import get_current_user
from api.models.user import User
from api.config import get_settings
from api.schemas.auth import (
    LoginRequest,
    SignupRequest,
    TokenResponse,
    UserResponse,
)
from security.jwt import create_access_token
from security.password import hash_password, verify_password

logger = structlog.get_logger()
router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()


def _token_response(email: str) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(subject=email),
        expires_in=settings.jwt_expire_minutes * 60,
    )


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(body: SignupRequest, db: AsyncSession = Depends(get_db)) -> TokenResponse:
    """Register a new user and return a JWT."""
    # Check for existing email
    result = await db.execute(select(User).where(User.email == body.email))
    if result.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
    )
    db.add(user)
    await db.flush()  # assigns id, commit happens in get_db dependency

    logger.info("user_signup", email=body.email, user_id=str(user.id))
    return _token_response(user.email)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)) -> TokenResponse:
    """Authenticate and return a JWT."""
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    logger.info("user_login", email=body.email)
    return _token_response(user.email)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)) -> UserResponse:
    """Return the authenticated user's profile. **Protected route.**"""
    return UserResponse.model_validate(current_user)
