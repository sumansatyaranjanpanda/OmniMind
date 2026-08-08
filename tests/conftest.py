"""Shared test fixtures for OmniMind tests.

Uses the Postgres instance from Docker Compose (TEST_DATABASE_URL env var
or defaults to the dev Postgres). Each test runs inside a transaction that
rolls back — so tests are isolated without needing to recreate the DB.

Tests that don't need the database (e.g., password, JWT) can run without
Docker by simply not requesting the ``client`` or ``db_session`` fixtures.
"""

import os
from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from api.database import Base, get_db
from api.models.user import User  # noqa: F401 — ensure model is registered

# ── Database URL ────────────────────────────────────────────────
# Use TEST_DATABASE_URL if set, otherwise fall back to dev Postgres.
TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://omnimind:omnimind@localhost:5433/omnimind_test",
)


from sqlalchemy import pool

# ── Engine & session factory for tests ──────────────────────────
test_engine = create_async_engine(TEST_DB_URL, echo=False, poolclass=pool.NullPool)
test_session_factory = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@pytest_asyncio.fixture(scope="session")
async def _db_tables():
    """Create all tables once at the start of the test session, drop at the end.

    This fixture is NOT autouse — it only runs when a test requests
    ``db_session`` or ``client``, which depend on it.
    """
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()


@pytest_asyncio.fixture
async def db_session(_db_tables: None) -> AsyncGenerator[AsyncSession, None]:
    """Yield a transactional session that rolls back after each test."""
    async with test_session_factory() as session, session.begin():
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Async test client with DB dependency overridden to use the test session."""
    from api.main import app

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db

    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
