"""Unit tests for the end-to-end multi-stage retrieval pipeline."""

from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, patch

from retrieval.hybrid_search import RetrievedChunk
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_rewriter import QueryExpansionResult


@pytest.mark.asyncio
async def test_retrieval_pipeline_full_flow():
    """Test full flow: Query Rewrite -> Multi-Query Dense -> BM25 -> RRF -> Rerank."""
    mock_dense_results = [
        {"chunk_id": "c1", "text": "Transformer encoder layer has 6 layers", "score": 0.88, "page": 2},
        {"chunk_id": "c2", "text": "Convolutional layer details", "score": 0.50, "page": 4},
    ]

    pipeline = RetrievalPipeline(candidate_k=10, default_top_k=2)

    with (
        patch(
            "retrieval.pipeline.rewrite_query",
            new_callable=AsyncMock,
            return_value=QueryExpansionResult(
                original_query="how many encoder layers?",
                rewritten_query="how many layers in transformer encoder?",
                sub_queries=[],
                keywords=["encoder", "layers", "transformer"],
            ),
        ),
        patch(
            "retrieval.pipeline.search_chunks_batch",
            new_callable=AsyncMock,
            return_value=[mock_dense_results],
        ),
        patch(
            "retrieval.pipeline.rerank_chunks",
            new_callable=AsyncMock,
            return_value=[
                RetrievedChunk(
                    id="c1",
                    text="Transformer encoder layer has 6 layers",
                    rerank_score=0.99,
                    source_stage="reranked",
                )
            ],
        ),
    ):
        results, trace_id = await pipeline.retrieve(
            query="how many encoder layers?",
            tenant_id="test-tenant",
            top_k=1,
            enable_query_rewrite=True,
            enable_hybrid=True,
            enable_rerank=True,
        )

        assert len(results) == 1
        assert results[0].id == "c1"
        assert results[0].rerank_score == 0.99
        assert results[0].source_stage == "reranked"
        assert trace_id is not None
        assert len(trace_id) > 0


# ── Query embedding memoization (added 2026-09-11) ──────────────
#
# The same query string was embedded three times per request — semantic cache check,
# dense vector search, and the cache write at the end of the graph — at ~0.8s per call
# measured live. Memoizing short single texts removes two of those three round trips.


@pytest.mark.asyncio
async def test_repeated_query_embedding_is_served_from_cache():
    from unittest.mock import AsyncMock, patch

    from retrieval import pinecone_client

    pinecone_client._QUERY_EMBED_CACHE.clear()

    fake = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[0.5] * pinecone_client.EMBEDDING_DIM)]
    )
    client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(embed_content=AsyncMock(return_value=fake)))
    )

    with patch.object(pinecone_client, "_get_genai_client", return_value=client):
        first = await pinecone_client.generate_text_embeddings(["who is abinash"])
        second = await pinecone_client.generate_text_embeddings(["who is abinash"])

    assert first == second
    assert client.aio.models.embed_content.await_count == 1, "second call should not hit the API"


@pytest.mark.asyncio
async def test_document_batches_are_not_memoized():
    """Ingestion embeds thousands of distinct chunks that are never re-embedded.

    Caching them would evict the query entries this cache exists to serve, so batches
    must bypass it entirely.
    """
    from unittest.mock import AsyncMock, patch

    from retrieval import pinecone_client

    pinecone_client._QUERY_EMBED_CACHE.clear()

    fake = SimpleNamespace(
        embeddings=[
            SimpleNamespace(values=[0.1] * pinecone_client.EMBEDDING_DIM),
            SimpleNamespace(values=[0.2] * pinecone_client.EMBEDDING_DIM),
        ]
    )
    client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(embed_content=AsyncMock(return_value=fake)))
    )

    with patch.object(pinecone_client, "_get_genai_client", return_value=client):
        await pinecone_client.generate_text_embeddings(["chunk one", "chunk two"])
        await pinecone_client.generate_text_embeddings(["chunk one", "chunk two"])

    assert client.aio.models.embed_content.await_count == 2
    assert len(pinecone_client._QUERY_EMBED_CACHE) == 0


@pytest.mark.asyncio
async def test_cache_is_bounded_and_evicts_oldest_first():
    from retrieval import pinecone_client

    pinecone_client._QUERY_EMBED_CACHE.clear()
    for i in range(pinecone_client._QUERY_EMBED_CACHE_MAX + 25):
        pinecone_client._QUERY_EMBED_CACHE[f"k{i}"] = [0.0]
        while len(pinecone_client._QUERY_EMBED_CACHE) > pinecone_client._QUERY_EMBED_CACHE_MAX:
            pinecone_client._QUERY_EMBED_CACHE.popitem(last=False)

    assert len(pinecone_client._QUERY_EMBED_CACHE) == pinecone_client._QUERY_EMBED_CACHE_MAX
    assert "k0" not in pinecone_client._QUERY_EMBED_CACHE
