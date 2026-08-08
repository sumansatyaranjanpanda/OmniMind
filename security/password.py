"""Password hashing and verification using bcrypt directly.

Keeps all password-related cryptography in one place. The rest of the app
never touches raw passwords — it calls hash_password() at signup and
verify_password() at login.

Note: We use bcrypt directly instead of passlib because passlib is unmaintained
and incompatible with bcrypt>=4.1.
"""

import bcrypt


def hash_password(plain: str) -> str:
    """Return a bcrypt hash of *plain*."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Check *plain* against a bcrypt *hashed* value."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
