"""FastAPI dependencies — reusable dependency callables injected into routes.

The main dependency here is ``get_current_user``, which extracts and validates
the JWT from the Authorization header, then looks up the user in the database.
"""

import uuid

import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from api.config import get_settings
from api.database import get_db
from api.models.user import User
from security.jwt import decode_access_token

logger = structlog.get_logger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)

DEMO_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
DEMO_USER_EMAIL = "demo@omnimind.ai"


async def get_or_create_demo_user(db: AsyncSession) -> User:
    """Ensure the fallback demo user exists in PostgreSQL to satisfy foreign key constraints."""
    try:
        result = await db.execute(
            select(User).where((User.id == DEMO_USER_ID) | (User.email == DEMO_USER_EMAIL))
        )
        user = result.scalar_one_or_none()
        if user is not None:
            return user

        demo_user = User(
            id=DEMO_USER_ID,
            email=DEMO_USER_EMAIL,
            hashed_password="$2b$12$dummyhashforlocaldemouserdevonly",
            is_active=True,
        )
        db.add(demo_user)
        await db.flush()
        return demo_user
    except Exception:
        return User(
            id=DEMO_USER_ID,
            email=DEMO_USER_EMAIL,
            hashed_password="",
            is_active=True,
        )


_UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Dependency: extract JWT → decode → fetch user, or reject with 401.

    This used to swallow every failure and return the built-in demo user, which meant
    a missing, expired, or forged token was indistinguishable from a valid one: every
    protected route answered 200 to anonymous callers, and because ``tenant_id`` comes
    from the resolved user, an anonymous caller was handed the demo tenant's documents.
    The demo fallback now only applies when it has been explicitly switched on for
    local use, and it is never reachable by default.
    """
    if credentials is not None:
        try:
            payload = decode_access_token(credentials.credentials)
            email: str | None = payload.get("sub")  # type: ignore[assignment]
        except JWTError:
            email = None
        if email:
            result = await db.execute(select(User).where(User.email == email))
            user = result.scalar_one_or_none()
            if user is not None and user.is_active:
                return user

    if get_settings().allow_demo_user_fallback:
        logger.warning(
            "Serving an unauthenticated request as the demo user — "
            "ALLOW_DEMO_USER_FALLBACK is enabled, do not use this outside local development",
            had_credentials=credentials is not None,
        )
        return await get_or_create_demo_user(db)

    raise _UNAUTHENTICATED
