"""Role-Based Access Control (RBAC) and Multi-Tenant Security Isolation.

Defines:
1. User Roles: ADMIN (full control), EDITOR (upload/index/read), VIEWER (read-only).
2. Permission Matrix: Read, Write, Delete, Eval, Admin.
3. Tenant Security Scoping: Enforces tenant isolation in vector search, graphs, and cache.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from fastapi import HTTPException, status
import structlog

logger = structlog.get_logger(__name__)


class UserRole(str, Enum):
    """User authorization roles."""

    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"


class DocumentSecurityLevel(str, Enum):
    """Classification levels for documents."""

    PUBLIC = "public"  # Accessible by all roles within tenant
    INTERNAL = "internal"  # Accessible by Member, Editor, Admin
    CONFIDENTIAL = "confidential"  # Accessible by Editor and Admin only
    RESTRICTED = "restricted"  # Accessible by Admin only


ROLE_PERMISSIONS: dict[UserRole, set[str]] = {
    UserRole.ADMIN: {"read", "write", "delete", "eval", "admin"},
    UserRole.EDITOR: {"read", "write", "eval"},
    UserRole.VIEWER: {"read"},
}

ROLE_DOC_ACCESS: dict[UserRole, set[DocumentSecurityLevel]] = {
    UserRole.ADMIN: {
        DocumentSecurityLevel.PUBLIC,
        DocumentSecurityLevel.INTERNAL,
        DocumentSecurityLevel.CONFIDENTIAL,
        DocumentSecurityLevel.RESTRICTED,
    },
    UserRole.EDITOR: {
        DocumentSecurityLevel.PUBLIC,
        DocumentSecurityLevel.INTERNAL,
        DocumentSecurityLevel.CONFIDENTIAL,
    },
    UserRole.VIEWER: {
        DocumentSecurityLevel.PUBLIC,
        DocumentSecurityLevel.INTERNAL,
    },
}


def check_permission(role: UserRole | str, required_permission: str) -> bool:
    """Check if a given role has the required permission."""
    try:
        user_role = role if isinstance(role, UserRole) else UserRole(str(role).lower())
    except ValueError:
        return False
    return required_permission in ROLE_PERMISSIONS.get(user_role, set())


def can_access_document(role: UserRole | str, doc_level: DocumentSecurityLevel | str) -> bool:
    """Check if a user role can access a document of a given security classification."""
    try:
        user_role = role if isinstance(role, UserRole) else UserRole(str(role).lower())
        level = doc_level if isinstance(doc_level, DocumentSecurityLevel) else DocumentSecurityLevel(str(doc_level).lower())
    except ValueError:
        return False
    return level in ROLE_DOC_ACCESS.get(user_role, set())


def get_tenant_metadata_filter(
    tenant_id: str,
    user_role: UserRole | str = UserRole.VIEWER,
) -> dict[str, Any]:
    """Generate Pinecone/Vector store metadata filter for tenant and role-based access control."""
    try:
        role = user_role if isinstance(user_role, UserRole) else UserRole(str(user_role).lower())
    except ValueError:
        role = UserRole.VIEWER

    allowed_levels = [lvl.value for lvl in ROLE_DOC_ACCESS.get(role, set())]

    return {
        "tenant_id": tenant_id,
        "security_level": {"$in": allowed_levels},
    }


def require_role(allowed_roles: list[UserRole]):
    """FastAPI dependency for role-based endpoint protection."""

    def role_checker(role: str = "viewer"):
        try:
            user_role = UserRole(role.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Invalid user role: {role}",
            )
        if user_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: Role '{role}' lacks required permissions.",
            )
        return user_role

    return role_checker
