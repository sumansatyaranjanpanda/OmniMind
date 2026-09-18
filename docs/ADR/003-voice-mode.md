# ADR 003 — Voice mode: Gemini Live API on a separate lane from the text graph

- **Status**: Accepted (implementation landed 2026-09-09)
- **Date**: 2026-09-09
- **Supersedes / amends**: nothing. Extends ADR 002 (Gemini-direct model access).

## Context

The text lane answers in roughly 20 seconds end to end. That is acceptable for a
typed answer and unusable for a spoken one — a voice agent that pauses 20 seconds
before replying is not a slow product, it is a broken one.

The latency is not incidental; it is where the quality comes from. `POST /chat`
runs about five sequential LLM round-trips: context rewriter → query analyzer →
retrieval → synthesizer → citation critic, plus up to two query-rewrite retries
when the critic's faithfulness gate (≥ 0.85) fails. Every one of those stages
exists for a reason, and the critic gate in particular is listed as
non-negotiable in `AGENTS.md`.

So the design problem is not "make the graph faster." It is: how do we add a
real-time spoken interface without either (a) shipping a fast agent that
confidently says wrong things, or (b) shipping a correct agent nobody will talk
to. Both failure modes lose the product.

## Decision

**Voice runs as a separate lane — a new `WS /voice/stream` endpoint backed by the
Gemini Live API (`gemini-3.1-flash-live-preview`), native audio in and out — not
as a variant of the text graph.**

Within that lane, three decisions do the actual work:

### 1. Latency is time-to-first-audio, not time-to-answer

A person says "let me check" and starts speaking in ~300ms, then delivers the
answer two seconds later, and the exchange feels fast. That reframing is the
unlock: expensive stages do not have to be *deleted*, they have to be *covered*
by speech. This is why voice mode did not have to trade away rigor.

### 2. Two tiers, routed by the model's own tool choice

The Live model is given two tools, and picking between them *is* the routing —
no separate router LLM, so routing costs nothing.

| Tier | Trigger | Path | Budget |
|---|---|---|---|
| 0 | Greetings, clarification, "say that again" | Live model answers in-session | ~400ms |
| 1 | Ordinary factual lookup (`search_documents`) | one hybrid retrieval pass, no LLM in the loop | ~1–1.5s |
| 2 | Comparison, synthesis, high-cost-if-wrong (`deep_research`) | **the full text agent graph, critic and faithfulness gate included** | ~3–5s, covered by an acknowledgment |

Tier 2 is the answer to "won't voice be lower quality?" For hard questions the
Live model is **the mouth, not the brain** — the existing critic-gated pipeline
produces the answer and the Live model only voices it. Nothing is given up.

### 3. The faithfulness gate moves upstream instead of being dropped

The citation critic cannot run inline in a voice turn: it grades an answer after
that answer exists, and a spoken sentence cannot be retracted. Rather than
lowering or skipping the gate, sufficiency is checked *before* the model speaks:

- `agents/voice/tools.py` applies an **evidence floor** (`VOICE_EVIDENCE_FLOOR`,
  default 0.35) to retrieval scores. Nothing above the floor → the tool returns
  `status="no_evidence"` and the system instruction requires the model to say it
  could not find the answer.
- The rationale is that most faithfulness failures trace back to thin or absent
  evidence, not to a model fabricating on top of good evidence. Catching the
  thin-evidence case upstream removes most of the failure mass at zero latency.
- The system instruction forbids answering document questions from general
  knowledge, and requires calling `search_documents` first.

This directly addresses the 2026-09-06 "who is abinash" incident recorded in
`AGENTS.md`: retrieval scoring, not post-hoc grading, is what prevents a
confident answer about the wrong subject.

## What voice reuses vs. what is new

**Reused unchanged**: Pinecone, `retrieval/pipeline.py` (same hybrid dense+BM25+RRF
+ rerank), `core/guardrails.sanitize_retrieved_content`, tenant scoping, the
conversation/message tables and `agents/memory/service.py` persistence.

**New**: `agents/voice/` (tools + Live session manager), `api/routers/voice.py`,
`frontend/src/services/voice.ts`, `frontend/src/components/Voice/`.

## Explicitly rejected alternatives

**Streaming cascade (STT → LLM → TTS, all streamed).** Three vendors, three
failure modes, and turn-taking/barge-in has to be built by hand. Native
speech-to-speech gets interruption, affective tone, and VAD from the API.

**Running the existing graph unchanged and just adding TTS.** This is the
20-second version with audio bolted on. Rejected as the thing the user explicitly
did not want.

**Dropping hybrid search to plain dense embeddings for speed.** Considered and
rejected on measurement grounds: BM25 + RRF is local and largely parallel, worth
a few hundred milliseconds, while the round-trips that actually cost seconds are
the LLM stages. Cutting it would have lost recall for almost no latency gain.
Query rewriting *is* disabled in Tier 1, because that one genuinely costs a full
LLM round-trip and the Live model already sends coreference-resolved queries.

**Token in the WebSocket URL query string.** Standard workaround for the browser
WebSocket API's inability to set headers, rejected because it writes a bearer
token into access logs, proxy logs, and browser history. Auth is a first-message
frame instead; nothing else is processed before it validates.

## Consequences

**Accepted costs**

- A second answer path to maintain. Tier 1 answers are generated by the Live
  model rather than by the synthesizer node, so a prompt change in one lane does
  not automatically apply to the other.
- Native-audio models reason less well than the text flagship. Mitigation is to
  widen Tier 2's routing, never to accept worse answers.
- Voice cost is per-minute (~$0.005 in / $0.018 out), not per-token.
- Audio is not persisted — only transcripts. Deliberate: storing raw voice
  recordings is a privacy and compliance surface nobody asked for.

**Ship criterion (not yet met)**

Voice must be evaluated, not vibe-checked. Run `evaluation/datasets/v1.jsonl`
through the voice retrieval path and compare faithfulness against the text lane.
Within a few points → ship. Otherwise widen Tier 2 routing or raise the evidence
floor. This also finally produces the RAGAS baseline that Phase 2's Definition of
Done has required since the 2026-08-28 reconciliation.

## Verified live (2026-09-10)

Tested end-to-end against a real Gemini Live session by streaming synthesized
speech (Windows SAPI, 16kHz mono PCM16 — the same format the browser mic
produces) through the actual `WS /voice/stream` protocol.

**Confirmed working:** grounded answers cited from the real corpus; multi-turn
conversation; coreference resolution ("what are *his* skills?" → tool query
`Abinash technical skills`, resolved in-session with no rewriter LLM call, which
is the bet this ADR made); noise robustness (~9dB added noise, no echo
cancellation — harder than real browser conditions); barge-in during a
`deep_research` call, with the tool task genuinely cancelled and the session
recovering to complete the next turn.

**Four bugs that only live testing could have caught:**

1. **Single-turn session.** `session.receive()` terminates at the end of *each*
   model turn (the SDK breaks on `turn_complete`). A single `async for` over it
   served one turn and then let the session tear down — voice answered the first
   question and silently hung up. Now wrapped in an outer re-arming loop.
2. **Evidence-gate false negative.** The identical fact scored 0.457 for the
   query `Abinash` and 0.259 for `Who is Abhinash` — the Live model's own
   (non-deterministic) phrasing, compounded by STT mishearing a proper noun.
   That returned "not in your documents" about a document sitting right there —
   precisely the failure class this ADR exists to prevent. Fixed with a
   question-wrapper-stripping retry before reporting no evidence.
3. **Uncancelled tool tasks on barge-in** — a slow `deep_research` kept running
   for an abandoned turn.
4. **Unflushed transcripts on interrupt** — the next turn's text concatenated
   onto the abandoned one, corrupting persisted history.

**Measured latency** (warm; the first call of a session is worse):

| Stage | Time |
|---|---|
| Query embedding (Gemini) | ~900ms |
| Pinecone dense query | ~850ms (5.2s cold) |
| BM25 + RRF | ~2ms |
| Cohere rerank | ~900ms |
| **Tier 1 retrieval total** | **~2.6s** |
| End-to-end time-to-first-audio | ~6.8s (13s on first turn) |

This misses the ~1-1.5s Tier-1 target in the table above. The ADR's premise
held — BM25/RRF really are free, and the round-trips really were the cost — but
the estimate assumed one network hop, whereas Tier 1 makes **three sequential
cloud calls to three vendors** (Google embed → Pinecone → Cohere). A significant
share is geographic: the index is in AWS us-east-1 and development is from India.
Options, in rough order of value: cache query embeddings in Redis; make Cohere
rerank optional for voice (~900ms, but it is what the evidence floor is
calibrated against, so this trades directly against gate accuracy); co-locate the
index; or move embedding local (new ADR territory).

Tier 0 conversational filler does cover this reasonably — the model says "let me
dig into that" within ~400ms — but the number should not be described as met.

## Verification

```powershell
pytest tests/unit/test_voice.py -v        # 27 passing as of 2026-09-10
cd frontend && npm run build              # type-checks the voice client + orb
```

Manual: `docker compose -f infrastructure/docker-compose.yml up -d`, start the
API, `cd frontend && npm run dev`, then click the waveform button in the chat
composer. Ask something answerable from an uploaded document, then something that
is not — the second must produce "I couldn't find that in your documents" rather
than a general-knowledge answer.
