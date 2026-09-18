"""Knowledge Graph Explorer API — endpoints for visual graph analytics and relationship discovery.

GET /graph               - Export full tenant graph as nodes/edges JSON
GET /graph/search        - Fuzzy search for entities
GET /graph/neighborhood  - Extract 1-to-N hop entity ego-subgraphs
GET /graph/paths         - Discover multi-hop paths between two entities
GET /graph/communities   - Group entities into high-level thematic clusters
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
import structlog

from api.deps import get_current_user
from api.models.user import User
from retrieval.graph_store import KnowledgeGraphStore

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/graph", tags=["knowledge_graph"])


@router.get("")
async def get_graph(
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict[str, Any]:
    """Retrieve full knowledge graph for the authenticated tenant."""
    tenant_id = str(current_user.id)
    raw_json = KnowledgeGraphStore.export_to_json(tenant_id)
    return json.loads(raw_json)


@router.get("/search")
async def search_graph_entities(
    current_user: Annotated[User, Depends(get_current_user)],
    q: str = Query(..., min_length=1, description="Entity search keyword"),
    top_k: int = Query(5, ge=1, le=20),
) -> list[dict[str, Any]]:
    """Find matching entities in the tenant knowledge graph."""
    tenant_id = str(current_user.id)
    return KnowledgeGraphStore.search_entities(tenant_id, q, top_k=top_k)


@router.get("/neighborhood")
async def get_entity_neighborhood(
    current_user: Annotated[User, Depends(get_current_user)],
    entity: str = Query(..., min_length=1, description="Target entity name"),
    max_hops: int = Query(2, ge=1, le=3),
) -> list[dict[str, Any]]:
    """Extract local subgraphs and connected relations around an entity."""
    tenant_id = str(current_user.id)
    return KnowledgeGraphStore.get_neighborhood(tenant_id, entity, max_hops=max_hops)


@router.get("/paths")
async def find_entity_paths(
    current_user: Annotated[User, Depends(get_current_user)],
    source: str = Query(..., min_length=1, description="Source entity name"),
    target: str = Query(..., min_length=1, description="Target entity name"),
    max_depth: int = Query(3, ge=1, le=5),
) -> list[list[str]]:
    """Find all multi-hop paths connecting two disparate entities."""
    tenant_id = str(current_user.id)
    return KnowledgeGraphStore.find_paths(tenant_id, source, target, max_depth=max_depth)


@router.get("/communities")
async def get_graph_communities(
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[dict[str, Any]]:
    """Detect thematic community clusters across the knowledge graph."""
    tenant_id = str(current_user.id)
    return KnowledgeGraphStore.get_community_summary(tenant_id)
