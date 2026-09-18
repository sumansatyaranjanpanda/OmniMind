---
title: OmniMind
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Enterprise RAG with citation-backed answers and voice mode
---

# OmniMind

Enterprise RAG/agent platform. Upload documents, ask questions, get answers where
every claim carries a citation back to the page it came from — plus a speech-to-speech
voice mode.

Source: https://github.com/sumansatyaranjanpanda/OmniMind

## What to try

1. Create an account (top right) — it is local to this Space.
2. Upload a PDF, DOCX, PPTX, XLSX, CSV, MD or HTML file and wait for it to finish
   embedding.
3. Ask something answerable only from that document. Every factual sentence should
   carry a `[^N]` marker you can click back to the source passage.
4. Ask something the document does not cover — it should say so and tell you the
   answer came from the web, rather than quietly answering anyway.
5. Try the voice button for a spoken conversation over the same corpus.

## How it works

```
query
  → input guardrail (injection/jailbreak block, PII mask)
  → semantic cache
  → context rewriter (coreference) ∥ episodic memory recall
  → query analysis ∥ hybrid retrieval (dense + BM25 + RRF + cross-encoder rerank)
  → evidence gate → source fusion → synthesis with inline citations
  → citation critic (faithfulness ≥ 0.85, retry loop on failure)
  → output guardrail (secret scrub)
```

## Known limits of this demo

- **Data is ephemeral.** Postgres runs inside the container, so accounts,
  conversations and document metadata reset whenever the Space restarts or wakes
  from sleep. Embedded vectors persist — those live in Pinecone.
- **The Space sleeps when idle.** The first request after a sleep pays a cold start
  while the container and its models come back up.
- **Shared API quota.** Every visitor's questions run against one set of API keys,
  so answers can slow down or fail with rate limits under load.
