"""JWT token creation and decoding.

Uses python-jose with HS256. Tokens carry a ``sub`` claim (the user's email)
and an ``exp`` claim.  Nothing else is stored in the token — user details
are fetched from the database on each request.
"""

from datetime import UTC, datetime, timedelta

from jose import JWTError, jwt

from api.config import get_settings

settings = get_settings()


def create_access_token(
    subject: str,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed JWT with *subject* as the ``sub`` claim."""
    expire = datetime.now(UTC) + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.jwt_expire_minutes)
    )
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, object]:
    """Decode and verify a JWT. Raises ``JWTError`` on failure."""
    try:
        return jwt.decode(  # type: ignore[no-any-return]
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except JWTError:
        raise
