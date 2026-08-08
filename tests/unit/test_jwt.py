"""Tests for JWT token creation and decoding."""

from datetime import timedelta

import pytest
from jose import JWTError

from security.jwt import create_access_token, decode_access_token


def test_create_and_decode_token():
    """A token created for a subject should decode back to that subject."""
    token = create_access_token(subject="user@example.com")
    payload = decode_access_token(token)
    assert payload["sub"] == "user@example.com"
    assert "exp" in payload


def test_token_with_custom_expiry():
    """Tokens should respect custom expiry deltas."""
    token = create_access_token(
        subject="user@example.com",
        expires_delta=timedelta(minutes=5),
    )
    payload = decode_access_token(token)
    assert payload["sub"] == "user@example.com"


def test_decode_invalid_token():
    """Decoding a garbage token should raise JWTError."""
    with pytest.raises(JWTError):
        decode_access_token("not-a-valid-token")


def test_decode_expired_token():
    """An expired token should raise JWTError."""
    token = create_access_token(
        subject="user@example.com",
        expires_delta=timedelta(seconds=-1),
    )
    with pytest.raises(JWTError):
        decode_access_token(token)
