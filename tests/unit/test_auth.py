"""Tests for auth endpoints — signup, login, and protected /me route."""

import pytest


@pytest.mark.asyncio
async def test_signup_creates_user_and_returns_token(client):
    """POST /auth/signup should create a user and return a JWT."""
    response = await client.post(
        "/auth/signup",
        json={"email": "new@example.com", "password": "strongpassword123"},
    )
    assert response.status_code == 201
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_signup_duplicate_email_returns_409(client):
    """Signing up with an already-registered email should return 409."""
    payload = {"email": "dupe@example.com", "password": "password123"}
    # First signup
    resp1 = await client.post("/auth/signup", json=payload)
    assert resp1.status_code == 201
    # Second signup with same email
    resp2 = await client.post("/auth/signup", json=payload)
    assert resp2.status_code == 409


@pytest.mark.asyncio
async def test_login_valid_credentials(client):
    """POST /auth/login with correct credentials should return a JWT."""
    # Signup first
    await client.post(
        "/auth/signup",
        json={"email": "login@example.com", "password": "mypassword"},
    )
    # Login
    response = await client.post(
        "/auth/login",
        json={"email": "login@example.com", "password": "mypassword"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data


@pytest.mark.asyncio
async def test_login_wrong_password(client):
    """POST /auth/login with wrong password should return 401."""
    await client.post(
        "/auth/signup",
        json={"email": "wrongpw@example.com", "password": "correct"},
    )
    response = await client.post(
        "/auth/login",
        json={"email": "wrongpw@example.com", "password": "incorrect"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_login_nonexistent_user(client):
    """POST /auth/login with an unregistered email should return 401."""
    response = await client.post(
        "/auth/login",
        json={"email": "nobody@example.com", "password": "whatever"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_me_with_valid_token(client):
    """GET /auth/me with a valid token should return user info."""
    # Signup to get a token
    signup_resp = await client.post(
        "/auth/signup",
        json={"email": "me@example.com", "password": "password123"},
    )
    token = signup_resp.json()["access_token"]

    # Access protected route
    response = await client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "me@example.com"
    assert data["is_active"] is True
    assert "id" in data


@pytest.mark.asyncio
async def test_me_without_token(client):
    """GET /auth/me without a token should return 401."""
    response = await client.get("/auth/me")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_me_with_invalid_token(client):
    """GET /auth/me with a garbage token should return 401."""
    response = await client.get(
        "/auth/me",
        headers={"Authorization": "Bearer invalid-token-here"},
    )
    assert response.status_code == 401
