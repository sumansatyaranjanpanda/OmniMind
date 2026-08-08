"""Tests for password hashing and verification."""

from security.password import hash_password, verify_password


def test_hash_password_returns_bcrypt_hash():
    """hash_password should return a bcrypt hash (starts with $2b$)."""
    hashed = hash_password("test-password")
    assert hashed.startswith("$2b$")
    assert hashed != "test-password"


def test_verify_password_correct():
    """verify_password should return True for the correct password."""
    hashed = hash_password("my-secret")
    assert verify_password("my-secret", hashed) is True


def test_verify_password_incorrect():
    """verify_password should return False for a wrong password."""
    hashed = hash_password("my-secret")
    assert verify_password("wrong-password", hashed) is False


def test_hash_password_unique_per_call():
    """Each call to hash_password should produce a different hash (unique salt)."""
    h1 = hash_password("same-password")
    h2 = hash_password("same-password")
    assert h1 != h2
