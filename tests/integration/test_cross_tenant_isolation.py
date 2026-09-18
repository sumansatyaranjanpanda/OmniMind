"""Cross-tenant isolation and entity-disambiguation checks — live Pinecone/Gemini.

Unlike tests/unit, these hit real external services and real already-ingested
documents, so they belong here rather than in the mocked unit suite. They exist because
two of this session's real incidents were both about the retriever handing the
synthesizer evidence about the WRONG subject:

1. Routing sent a person-question to the graph/web engines instead of the tenant's own
   documents ("who is abinash" answered from four unrelated public figures).
2. The fix for (1) raises a related question that had never actually been tested: if a
   tenant's OWN corpus contains two different people who happen to share a name, does
   the synthesizer keep them apart, or silently blend their facts together?

And separately: multi-tenant SaaS with per-tenant Pinecone namespaces has an obvious
failure mode if namespace isolation has any gap — one tenant's query returning another
tenant's private data. Nobody had verified this against two real tenants that both
happen to contain a document.

Requires: PINECONE_API_KEY/PINECONE_INDEX_HOST and GEMINI_API_KEY set, and the two
fixture tenant IDs below actually ingested. Skips cleanly if either is unavailable —
this is a smoke check against real data, not something CI should block on.

    pytest tests/integration/test_cross_tenant_isolation.py -v
"""

from __future__ import annotations

import pytest

from api.config import get_settings

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
async def _fresh_pinecone_client_per_test():
    """Reset the module-level async Pinecone client between tests.

    pytest-asyncio gives each test function its own event loop by default
    (asyncio_mode="auto" with the default function-scoped loop). retrieval.
    pinecone_client caches its AsyncIndex as a module-level singleton the first time
    it's used, and that client's underlying transport is bound to whichever loop was
    running at creation time — so the second test function in this file inherited a
    client wired to the FIRST test's already-closed loop and every Pinecone call in it
    failed with "Event loop is closed". This has no production equivalent (a real
    server has exactly one long-lived loop), so the fix belongs here, not in the
    application code: force a fresh client per test.
    """
    import retrieval.pinecone_client as pc_mod

    # Reset before AND after: something earlier in the same pytest run (a unit test
    # importing pinecone_client, or a prior test in this file) can leave a client bound
    # to an event loop that's since been closed, and that stale state is what actually
    # causes the failure — not just what this test itself might leave behind.
    await pc_mod.close_async_index()
    yield
    await pc_mod.close_async_index()

# Two tenants already used as this project's real test fixtures (see AGENTS.md's
# testing notes). Swap these for your own tenant UUIDs if these documents are gone.
RESUME_TENANT = "6e64db7c-7133-4211-be2e-85e951aae305"
NORTHWIND_TENANT = "feb3eee1-3163-442f-a764-d5103cdbe6bd"


def _require_live_services() -> None:
    settings = get_settings()
    if not settings.pinecone_api_key or not settings.gemini_api_key:
        pytest.skip("PINECONE_API_KEY / GEMINI_API_KEY not configured")


# ── Cross-tenant leakage ─────────────────────────────────────────


@pytest.mark.parametrize(
    "tenant,query,forbidden_terms",
    [
        (RESUME_TENANT, "Q3 revenue business review", ["Northwind", "Robotics", "FY2026"]),
        (RESUME_TENANT, "SLA uptime guarantee price book", ["Northwind", "SLA", "Price Book"]),
        (
            NORTHWIND_TENANT,
            "who is abinash software engineer",
            ["Abinash", "Thakur", "Radical Minds", "Antino"],
        ),
        (NORTHWIND_TENANT, "leetcode contest rating skills", ["Abinash", "LeetCode", "1577"]),
    ],
)
async def test_no_cross_tenant_leakage(tenant, query, forbidden_terms) -> None:
    """A tenant's retrieval must never surface another tenant's chunks.

    Pinecone namespaces are the only thing enforcing this. If a namespace filter is
    ever dropped from a query path, this is silent in every other test — those all
    query within one tenant and would never notice a different tenant's data mixed in.
    """
    _require_live_services()
    from retrieval.pipeline import execute_retrieval

    chunks, _ = await execute_retrieval(
        query=query,
        tenant_id=tenant,
        top_k=10,
        enable_query_rewrite=False,
        enable_hybrid=True,
        enable_rerank=True,
    )

    leaked = [
        c for c in chunks if any(term.lower() in c.text.lower() for term in forbidden_terms)
    ]
    assert not leaked, (
        f"tenant {tenant} leaked {len(leaked)} chunk(s) containing another tenant's terms: "
        + "; ".join(c.text[:100] for c in leaked)
    )


# ── Same-name entity disambiguation ─────────────────────────────


async def test_synthesizer_separates_two_people_with_the_same_name() -> None:
    """Two different people sharing a first name, in the SAME tenant's evidence.

    The synthesis prompt (rule 7) says never to describe a different entity that
    merely shares a name — but until now that had only been tested against an
    obviously-unrelated web stranger, an easy case. This tests the harder one: two
    genuinely similar internal documents (same name, same job family, overlapping
    numeric facts) where blending is the tempting shortcut for the model to take.
    """
    _require_live_services()
    from agents.nodes.synthesizer import synthesizer_node

    fused_evidence = [
        {
            "source_id": 1,
            "marker": "[^1]",
            "source_type": "document",
            "title": "AbinashResume.pdf | Achievements",
            "content": "Abinash Thakur solved 1000+ problems on LeetCode with a contest rating of 1577.",
        },
        {
            "source_id": 2,
            "marker": "[^2]",
            "source_type": "document",
            "title": "InternProfiles.pdf | Backend Team",
            "content": (
                "Abinash Reddy, backend intern, has a LeetCode contest rating of 2100 "
                "and focuses on competitive programming."
            ),
        },
    ]

    out = await synthesizer_node(
        {
            "query": "what is abinash's contest rating",
            "raw_query": "what is abinash's contest rating",
            "fused_evidence": fused_evidence,
            "route_history": [],
            "chat_history": [],
        }
    )
    answer = out["draft_answer"]
    lower = answer.lower()

    assert "1577" in answer and "2100" in answer, (
        f"expected both ratings to appear distinctly, got: {answer}"
    )
    assert "thakur" in lower and "reddy" in lower, "both people must be named, not merged into one"

    # Attribution is checked per structural unit (line/bullet), not by raw character
    # distance — a short bullet list can put the next person's name closer in
    # character count to this bullet's number than that number's own subject is, which
    # makes distance-in-characters an unreliable proxy for "who does this number belong
    # to". Splitting on newlines and requiring each rating's line contain the RIGHT
    # name and not the WRONG one is what actually tests attribution.
    lines = [ln for ln in lower.splitlines() if ln.strip()]
    line_with_1577 = next((ln for ln in lines if "1577" in ln), None)
    line_with_2100 = next((ln for ln in lines if "2100" in ln), None)
    assert line_with_1577 is not None and line_with_2100 is not None

    assert "thakur" in line_with_1577 and "reddy" not in line_with_1577, (
        f"1577 (Thakur's rating) not attributed to Thakur alone: {line_with_1577!r}"
    )
    assert "reddy" in line_with_2100 and "thakur" not in line_with_2100, (
        f"2100 (Reddy's rating) not attributed to Reddy alone: {line_with_2100!r}"
    )
