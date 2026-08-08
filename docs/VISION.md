# OmniMind — Vision

## What OmniMind will become

OmniMind is an enterprise RAG/agent platform designed to:

1. **Ingest** documents of any format (PDF, DOCX, PPTX, XLSX, CSV, Markdown, HTML,
   images with OCR) while preserving structure (headings, tables, figures)
2. **Retrieve** relevant content using hybrid search — dense embeddings + sparse BM25
   vectors in a single Pinecone index, with cross-encoder reranking
3. **Reason** over evidence using a multi-agent LangGraph pipeline with a supervisor
   coordinating specialized agents (SQL agent, web agent, knowledge graph agent)
4. **Answer** with citations — every claim traces back to a specific page, section, and
   source document, or the system explicitly states "insufficient evidence"

## Phased Roadmap

| Phase | Focus | Key Deliverables |
|---|---|---|
| 1 | Foundation | Monorepo, FastAPI, Postgres, Redis, MinIO, auth, Docker Compose, CI |
| 2 | Ingestion & Search | Multi-format ingestion, structure-aware parsing, chunking, embeddings, Pinecone, citations |
| 3 | Hybrid Search | BM25 + dense fusion, cross-encoder reranking, query rewriting |
| 4 | Multimodal | Image/table/OCR extraction and search |
| 5 | Agents | LangGraph supervisor, SQL agent, web agent, planner, critic |
| 6 | Knowledge Graph | Entity extraction, graph construction, graph-augmented retrieval |
| 7 | Evaluation | RAGAS harness, regression tracking, A/B eval framework |
| 8 | Multi-tenancy | Tenant isolation, RBAC, namespace-scoped search |
| 9 | Production | Kubernetes, auto-scaling, cost/latency optimization |

## Design Principles

- **Ship in phases** — each phase is self-contained and useful on its own
- **Typed everywhere** — Pydantic models at boundaries, typed functions, mypy strict
- **Test with the code** — tests land alongside the implementation
- **Security by default** — document content is untrusted, LLM SQL is read-only
- **Observable** — structured logging + Langfuse tracing from Phase 2
- **Provider-agnostic** — OpenRouter as LLM gateway, no vendor lock-in in business logic
