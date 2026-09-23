"""Managed-provider database URL normalization (api/config.py).

Render, Heroku and Railway hand out `postgresql://…` (Heroku still emits the legacy
`postgres://`). SQLAlchemy maps both to the default SYNCHRONOUS psycopg2 driver, and
create_async_engine then rejects them at import time — which on a managed host looks
like an unexplained boot loop rather than a readable error. Caught before deploying
to Render, not after.
"""

import pytest

from api.config import Settings


def _url(value: str) -> str:
    return Settings(database_url=value).database_url


def test_render_style_url_is_upgraded_to_asyncpg():
    got = _url("postgresql://omnimind:secret@dpg-abc123.singapore-postgres.render.com/omnimind")
    assert got.startswith("postgresql+asyncpg://")


def test_heroku_legacy_postgres_scheme_is_handled():
    """Heroku still emits the deprecated `postgres://` spelling."""
    got = _url("postgres://user:pw@host:5432/db")
    assert got.startswith("postgresql+asyncpg://")
    assert "user:pw@host:5432/db" in got


def test_an_already_async_url_is_left_alone():
    original = "postgresql+asyncpg://omnimind:omnimind@localhost:5432/omnimind"
    assert _url(original) == original


def test_credentials_and_query_params_survive_normalization():
    """sslmode=require is mandatory on several managed providers — dropping it
    would turn a working connection string into a refused connection."""
    got = _url("postgresql://u:p%40ss@host:5432/db?sslmode=require")
    assert got == "postgresql+asyncpg://u:p%40ss@host:5432/db?sslmode=require"


@pytest.mark.parametrize("scheme", ["postgresql+psycopg", "postgresql+asyncpg"])
def test_explicit_driver_choices_are_respected(scheme):
    """If someone deliberately picked a driver, don't override it."""
    original = f"{scheme}://user:pw@host/db"
    assert _url(original) == original
