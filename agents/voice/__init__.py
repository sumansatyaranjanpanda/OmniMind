"""Voice mode — real-time speech-to-speech over the Gemini Live API.

Deliberately a *separate lane* from the text agent graph, not a variant of it.
See docs/ADR/003-voice-mode.md for the reasoning; the short version is that the
text graph's quality comes from ~5 sequential LLM round-trips (analyzer →
retrieve → fuse → synthesize → critic → optional rewrite loop), which is
correct for a typed answer and unusable for a spoken turn.

Voice keeps the quality-critical parts and drops only the ones the Live model
already performs itself:

- coreference resolution  → the Live model has the whole session in context
- intent routing          → the Live model's choice of tool *is* the routing
- synthesis               → the Live model speaks directly from tool evidence
- faithfulness gating     → moved *upstream* into an evidence-sufficiency floor
                            (see ``tools.py``), because a post-hoc critic cannot
                            un-speak a sentence that was already said out loud

Retrieval itself is NOT downgraded: voice runs the same hybrid dense+BM25+RRF
pipeline the text lane runs, because retrieval is a few hundred milliseconds
and mostly parallel — the latency was never there.
"""

from agents.voice.tools import VoiceToolbox, build_function_declarations

__all__ = ["VoiceToolbox", "build_function_declarations"]
