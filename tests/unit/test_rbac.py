"""Unit tests for Phase 6 RBAC and Tenant Isolation Security."""

from __future__ import annotations

import pytest

from security.rbac import (
    DocumentSecurityLevel,
    UserRole,
    can_access_document,
    check_permission,
    get_tenant_metadata_filter,
)


def test_role_permissions():
    """Test permission matrix across user roles."""
    # Admin has all permissions
    assert check_permission(UserRole.ADMIN, "read") is True
    assert check_permission(UserRole.ADMIN, "write") is True
    assert check_permission(UserRole.ADMIN, "delete") is True
    assert check_permission(UserRole.ADMIN, "eval") is True
    assert check_permission(UserRole.ADMIN, "admin") is True

    # Editor has read, write, eval
    assert check_permission(UserRole.EDITOR, "read") is True
    assert check_permission(UserRole.EDITOR, "write") is True
    assert check_permission(UserRole.EDITOR, "delete") is False

    # Viewer has read only
    assert check_permission(UserRole.VIEWER, "read") is True
    assert check_permission(UserRole.VIEWER, "write") is False
    assert check_permission(UserRole.VIEWER, "delete") is False


def test_document_security_access():
    """Test document-level classification access."""
    # Viewer can access public and internal only
    assert can_access_document(UserRole.VIEWER, DocumentSecurityLevel.PUBLIC) is True
    assert can_access_document(UserRole.VIEWER, DocumentSecurityLevel.INTERNAL) is True
    assert can_access_document(UserRole.VIEWER, DocumentSecurityLevel.CONFIDENTIAL) is False
    assert can_access_document(UserRole.VIEWER, DocumentSecurityLevel.RESTRICTED) is False

    # Editor can access confidential
    assert can_access_document(UserRole.EDITOR, DocumentSecurityLevel.CONFIDENTIAL) is True
    assert can_access_document(UserRole.EDITOR, DocumentSecurityLevel.RESTRICTED) is False

    # Admin can access restricted
    assert can_access_document(UserRole.ADMIN, DocumentSecurityLevel.RESTRICTED) is True


def test_tenant_metadata_filter_generation():
    """Test metadata filters generated for Pinecone vector queries."""
    tenant = "tenant_enterprise_99"

    # Viewer filter excludes confidential & restricted
    viewer_filter = get_tenant_metadata_filter(tenant, UserRole.VIEWER)
    assert viewer_filter["tenant_id"] == tenant
    assert "public" in viewer_filter["security_level"]["$in"]
    assert "internal" in viewer_filter["security_level"]["$in"]
    assert "restricted" not in viewer_filter["security_level"]["$in"]

    # Admin filter includes all security levels
    admin_filter = get_tenant_metadata_filter(tenant, UserRole.ADMIN)
    assert admin_filter["tenant_id"] == tenant
    assert "restricted" in admin_filter["security_level"]["$in"]
    assert "confidential" in admin_filter["security_level"]["$in"]
