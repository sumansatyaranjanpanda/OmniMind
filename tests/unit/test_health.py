"""Tests for the health-check endpoint."""

import pytest


@pytest.mark.asyncio
async def test_health_returns_200(client):
    """GET /health should return 200 with a status field."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data
