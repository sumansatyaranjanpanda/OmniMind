"""Gemini Live session manager — bridges a browser WebSocket to the Live API.

Two independent pumps run concurrently for the lifetime of a call:

    uplink    browser mic PCM ──► Live API
    downlink  Live API audio  ──► browser speaker, plus transcript/state events

They are separate tasks rather than one loop because they must not block each
other. If the uplink waited on the downlink, the user could not interrupt the
model mid-sentence — barge-in is the single feature that makes a voice agent
feel real rather than like a walkie-talkie.

For the same reason tool calls are dispatched as their own tasks and their
results sent when they finish. A ``deep_research`` call takes several seconds;
awaiting it inline inside the receive loop would freeze audio playback and
ignore the user's voice for that whole window.

Audio format is fixed by the API: 16 kHz signed 16-bit mono PCM inbound,
24 kHz signed 16-bit mono PCM outbound.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from google.genai import types

from agents.llm_helper import get_genai_client
from agents.voice.tools import VoiceToolbox, build_function_declarations
from api.config import get_settings

logger = structlog.get_logger(__name__)

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000

EmitFn = Callable[[str, dict[str, Any]], Awaitable[None]]

# Prebuilt Live API voices offered in the picker, with the character each one
# is documented to convey. This is a property of the API rather than a
# deployment tunable, so it lives here instead of in Settings — but the
# *default* still comes from config (``voice_name``).
#
# An allowlist rather than passing the client's string through: the voice name
# goes straight into the Live session config, and an unrecognised value fails
# the connection *after* the WebSocket is already open, which surfaces to the
# user as a voice call that dies on connect for no visible reason.
AVAILABLE_VOICES: dict[str, str] = {
    "Puck": "Upbeat",
    "Charon": "Informative",
    "Kore": "Firm",
    "Fenrir": "Excitable",
    "Aoede": "Breezy",
    "Leda": "Youthful",
    "Orus": "Warm",
    "Zephyr": "Bright",
}


def resolve_voice(requested: str | None) -> str:
    """Validated voice name, falling back to the configured default."""
    if requested and requested in AVAILABLE_VOICES:
        return requested
    return get_settings().voice_name


SYSTEM_INSTRUCTION = """You are OmniMind, a voice assistant that answers strictly from the user's own uploaded documents.

HOW YOU SOUND
You are speaking out loud, not writing. Keep answers to one or two sentences unless asked to elaborate. Use contractions. Never read out markdown, bullet points, URLs, or citation markers like [^1] — they are unspeakable. Numbers and dates should be said the way a person would say them.

GROUNDING — THIS IS NOT OPTIONAL
The user's documents are the source of truth, and you cannot see them without calling a tool. So:
- Call search_documents before answering ANY question about the user's files, data, projects, or any person, place, or thing that might appear in their documents.
- Never answer such a question from your own knowledge. You do not know what is in their documents until you look.
- If a tool returns status "no_evidence", say plainly that you could not find it in their documents. Do not fill the gap with a guess, a general-knowledge answer, or a plausible-sounding invention. "I couldn't find anything about that in your documents" is a correct and complete answer.
- When you do answer from evidence, name the source once, conversationally: "your résumé says...", "according to the architecture doc...".

CHOOSING A TOOL
- search_documents for nearly everything. It is fast.
- deep_research only for questions needing comparison across documents, multi-step reasoning, or where a wrong answer would be costly. It takes a few seconds, so ALWAYS say something first — "let me dig into that" — then call it. Never sit in silence waiting on a tool.

You may answer without any tool ONLY for conversational turns that are not about content: greetings, thanks, "can you repeat that", "speak slower", or clarifying what the user meant.
"""


class VoiceSession:
    """One authenticated speech-to-speech call."""

    def __init__(self, toolbox: VoiceToolbox, emit: EmitFn, voice: str | None = None) -> None:
        self.toolbox = toolbox
        self.emit = emit
        self.voice = resolve_voice(voice)
        self._session: Any = None
        self._tool_tasks: set[asyncio.Task[None]] = set()

    # ── Uplink: browser → Live ──────────────────────────────────

    async def send_audio(self, pcm: bytes) -> None:
        """Forward a mic chunk. Server-side VAD handles turn detection."""
        if self._session is None:
            return
        await self._session.send_realtime_input(
            audio=types.Blob(data=pcm, mime_type=f"audio/pcm;rate={INPUT_SAMPLE_RATE}")
        )

    async def send_text(self, text: str) -> None:
        """Inject a typed message into the live conversation."""
        if self._session is None:
            return
        await self._session.send_realtime_input(text=text)

    # ── Downlink: Live → browser ────────────────────────────────

    async def _run_tool(self, call: Any) -> None:
        """Execute one tool call and return its result, off the receive loop."""
        args = dict(call.args or {})
        await self.emit("tool_start", {"tool": call.name, "query": args.get("query", "")})

        result = await self.toolbox.dispatch(call.name, args)

        await self.emit(
            "tool_end",
            {
                "tool": call.name,
                "status": result.get("status", "ok"),
                "citations": self.toolbox.citations,
            },
        )

        if self._session is None:
            return
        try:
            await self._session.send_tool_response(
                function_responses=[
                    types.FunctionResponse(id=call.id, name=call.name, response=result)
                ]
            )
        except Exception as exc:
            # The call may have been cancelled by a barge-in while the tool ran —
            # the session is gone but the conversation isn't broken, so log and move on.
            logger.warning("Could not deliver tool response", tool=call.name, error=str(exc))

    def _spawn_tool(self, call: Any) -> None:
        task = asyncio.create_task(self._run_tool(call))
        # Held in a set so the task isn't garbage-collected mid-flight; asyncio
        # only keeps a weak reference to running tasks.
        self._tool_tasks.add(task)
        task.add_done_callback(self._tool_tasks.discard)

    async def _receive_loop(self) -> None:
        """Pump Live messages for the whole call, across every turn.

        ``session.receive()`` deliberately terminates at the end of EACH model
        turn — the SDK breaks out of its own loop on ``turn_complete``, and its
        docstring says the returned responses "represent a complete model turn".
        So a single ``async for`` over it handles exactly one turn and then falls
        off the end, which finishes this task and tears down the session.

        Verified live 2026-09-10: that made voice mode single-turn. The first
        answer arrived correctly, then the connection closed and the user could
        not say anything else — the failure looked like a mysterious silent
        disconnect rather than a missing loop. The outer loop re-arms receive()
        for the next turn.
        """
        assert self._session is not None
        while True:
            turn_had_messages = False
            async for response in self._session.receive():
                turn_had_messages = True
                await self._handle_response(response)

            if not turn_had_messages:
                # receive() yielding nothing at all means the underlying
                # connection is finished, not that a turn ended. Without this
                # the outer loop would spin hot against a dead session.
                logger.info("Live connection closed — receive loop exiting")
                return

    async def _handle_response(self, response: Any) -> None:
        """Fan one Live message out to the browser."""
        # Raw PCM for the speaker. Sent as its own binary frame rather than
        # base64 inside JSON — base64 inflates every audio chunk by 33% on
        # the hot path, which is real latency at 24 kHz.
        if response.data:
            await self.emit("__binary__", {"pcm": response.data})

        server_content = response.server_content
        if server_content is not None:
            if server_content.interrupted:
                # The user started talking over the model. The browser must
                # drop whatever is queued for playback immediately, or it
                # keeps speaking a reply the user already interrupted.
                await self.emit("interrupted", {})

                # Anything still in flight belongs to the turn being cut off —
                # a new turn can't have started a tool call yet, since that
                # only happens after the model processes the next input, which
                # comes after this interruption. Left uncancelled, a slow
                # deep_research call keeps running for a turn nobody is
                # waiting on and later delivers send_tool_response for an
                # abandoned turn.
                for task in list(self._tool_tasks):
                    task.cancel()

            if server_content.input_transcription is not None:
                text = server_content.input_transcription.text
                if text:
                    await self.emit("user_transcript", {"text": text})

            if server_content.output_transcription is not None:
                text = server_content.output_transcription.text
                if text:
                    await self.emit("agent_transcript", {"text": text})

            if server_content.turn_complete:
                await self.emit("turn_complete", {"citations": self.toolbox.citations})

        if response.tool_call is not None:
            for call in response.tool_call.function_calls or []:
                self._spawn_tool(call)

        if response.tool_call_cancellation is not None:
            logger.info(
                "Live server cancelled tool call(s)",
                ids=response.tool_call_cancellation.ids,
            )

        if response.go_away is not None:
            logger.warning(
                "Live server sent go_away — connection ending soon",
                time_left=response.go_away.time_left,
            )

    # ── Lifecycle ───────────────────────────────────────────────

    async def run(self, inbound: Callable[[], Awaitable[dict[str, Any] | None]]) -> None:
        """Open the Live connection and pump both directions until the call ends.

        ``inbound`` is awaited repeatedly for the next browser message and
        should return None when the client disconnects.
        """
        settings = get_settings()
        client = get_genai_client()
        if client is None:
            await self.emit("error", {"error": "Voice is unavailable — GEMINI_API_KEY is not configured."})
            return

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(parts=[types.Part(text=SYSTEM_INSTRUCTION)]),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice)
                )
            ),
            tools=[types.Tool(function_declarations=build_function_declarations())],
            # Transcriptions drive the on-screen transcript. Without these the
            # UI would have audio and nothing readable to show alongside it.
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )

        try:
            async with client.aio.live.connect(
                model=settings.voice_live_model, config=config
            ) as session:
                self._session = session
                await self.emit("ready", {"model": settings.voice_live_model, "voice": self.voice})

                receiver = asyncio.create_task(self._receive_loop())
                try:
                    while True:
                        message = await inbound()
                        if message is None:
                            break
                        if receiver.done():
                            break

                        if "pcm" in message:
                            await self.send_audio(message["pcm"])
                        elif message.get("type") == "text":
                            await self.send_text(str(message.get("text", "")))
                finally:
                    receiver.cancel()
                    for task in list(self._tool_tasks):
                        task.cancel()
                    await asyncio.gather(receiver, *self._tool_tasks, return_exceptions=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Live session failed", error=str(exc))
            await self.emit("error", {"error": "The voice connection dropped."})
        finally:
            self._session = None
