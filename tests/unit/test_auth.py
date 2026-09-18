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
        json={"email": "login@example.com", "password": "mypassword1"},
    )
    # Login
    response = await client.post(
        "/auth/login",
        json={"email": "login@example.com", "password": "mypassword1"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data


@pytest.mark.asyncio
async def test_login_wrong_password(client):
    """POST /auth/login with wrong password should return 401."""
    await client.post(
        "/auth/signup",
        json={"email": "wrongpw@example.com", "password": "correct123"},
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


# ── Password policy (api/schemas/auth.py) ───────────────────────


@pytest.mark.parametrize(
    "password,reason",
    [
        ("shrt1", "shorter than the 8-character minimum"),
        ("nodigitshere", "no digit"),
        ("12345678", "no letter"),
        ("a1" + "x" * 71, "longer than bcrypt's 72-byte input limit"),
    ],
)
@pytest.mark.asyncio
async def test_signup_rejects_weak_password(client, password, reason):
    """Signup must enforce the policy server-side, whatever the client sends.

    bcrypt hashes only the first 72 bytes and silently ignores the rest, so an
    over-long passphrase is no stronger than its prefix — reject it rather than
    accept a password we only partly check.
    """
    response = await client.post(
        "/auth/signup",
        json={"email": f"weak-{len(password)}-{reason[:4]}@example.com", "password": password},
    )
    assert response.status_code == 422, f"should reject: {reason}"


@pytest.mark.asyncio
async def test_signup_normalises_email_case(client):
    """One person must not be able to register the same address twice.

    The uniqueness check is an equality match on the stored column, so without
    normalisation "User@Example.com" and "user@example.com" become two accounts
    and which one you log into depends on how you typed it.
    """
    first = await client.post(
        "/auth/signup",
        json={"email": "MixedCase@Example.com", "password": "validpass1"},
    )
    assert first.status_code == 201

    duplicate = await client.post(
        "/auth/signup",
        json={"email": "mixedcase@example.com", "password": "validpass1"},
    )
    assert duplicate.status_code == 409

    # ...and the original casing still logs in.
    login = await client.post(
        "/auth/login",
        json={"email": "MIXEDCASE@example.com", "password": "validpass1"},
    )
    assert login.status_code == 200


@pytest.mark.asyncio
async def test_token_response_reports_expiry(client):
    """The client needs to know when the token dies, rather than discovering it
    from a failed action halfway through a task."""
    response = await client.post(
        "/auth/signup",
        json={"email": "expiry@example.com", "password": "validpass1"},
    )
    assert response.status_code == 201
    assert response.json()["expires_in"] > 0
