"""Voice mode WebSocket — real-time speech-to-speech.

    WS /voice/stream

Protocol, browser side:
    1. connect, then send ONE JSON frame: {"token": "<jwt>", "thread_id": "<uuid|null>"}
    2. thereafter send binary frames of 16 kHz mono PCM16 mic audio
    3. receive binary frames of 24 kHz mono PCM16 speaker audio, interleaved with
       JSON frames for transcript/state events

Auth is a first-message frame rather than the usual ``Depends(get_current_user)``
because the browser WebSocket API cannot set an Authorization header. The obvious
workaround — ``?token=...`` in the URL — puts a bearer token into access logs,
proxy logs, and browser history, so it is avoided here: the socket is accepted,
then closed with 1008 if the first frame does not carry a valid token. Nothing
but that first frame is processed before the token is validated.

Voice turns are persisted into the same conversation tables as typed chat, so a
call and a chat thread are one continuous history rather than two disconnected
records.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from jose import JWTError
from sqlalchemy import select

from agents.memory.service import load_conversation_context, persist_turn_background
from agents.voice.session import AVAILABLE_VOICES, VoiceSession
from agents.voice.tools import VoiceToolbox
from api.config import get_settings
from api.database import async_session_factory
from api.models.user import User
from security.jwt import decode_access_token

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])

# A caller that connects and never authenticates would otherwise hold a socket
# and an event-loop task open indefinitely.
AUTH_TIMEOUT_SECONDS = 10.0

WS_POLICY_VIOLATION = 1008
WS_INTERNAL_ERROR = 1011


async def _authenticate(websocket: WebSocket) -> tuple[User, str | None, str | None] | None:
    """Read and validate the opening auth frame. Returns (user, thread_id, voice) or None."""
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=AUTH_TIMEOUT_SECONDS)
    except (TimeoutError, WebSocketDisconnect):
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None

    token = payload.get("token")
    if not isinstance(token, str) or not token:
        return None

    try:
        claims = decode_access_token(token)
        email = claims.get("sub")
    except JWTError:
        return None

    if not email:
        return None

    async with async_session_factory() as db:
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

    if user is None or not user.is_active:
        return None

    thread_id = payload.get("thread_id")
    voice = payload.get("voice")
    return (
        user,
        thread_id if isinstance(thread_id, str) else None,
        voice if isinstance(voice, str) else None,
    )


@router.get("/voices")
async def list_voices() -> dict[str, Any]:
    """Voice catalogue for the picker.

    Served from the server so the list has one source of truth — a hard-coded
    copy in the frontend would silently drift from the allowlist that actually
    validates the choice, and the mismatch would only show up as a call that
    dies on connect.
    """
    settings = get_settings()
    return {
        "default": settings.voice_name,
        "voices": [{"name": name, "character": character} for name, character in AVAILABLE_VOICES.items()],
    }


@router.websocket("/stream")
async def voice_stream(websocket: WebSocket) -> None:
    """Bridge the browser's microphone to a Gemini Live session."""
    settings = get_settings()
    await websocket.accept()

    if not settings.voice_enabled:
        await websocket.close(code=WS_POLICY_VIOLATION, reason="Voice mode is disabled")
        return

    auth = await _authenticate(websocket)
    if auth is None:
        await websocket.close(code=WS_POLICY_VIOLATION, reason="Not authenticated")
        return

    user, requested_thread_id, requested_voice = auth
    user_id = str(user.id)

    # Resolve (or create) the conversation up front so voice turns land in the
    # same thread the text UI reads back.
    async with async_session_factory() as db:
        ctx = await load_conversation_context(db, requested_thread_id, user.id)
        thread_id = str(ctx.conversation.id)

    toolbox = VoiceToolbox(tenant_id=user_id, user_id=user_id, thread_id=thread_id)

    # Transcript fragments arrive token-by-token; they're accumulated per turn so
    # a completed turn can be written to history as one user/assistant exchange.
    pending_user = ""
    pending_agent = ""

    async def flush_pending_turn() -> None:
        nonlocal pending_user, pending_agent
        if pending_user.strip() and pending_agent.strip():
            await persist_turn_background(
                thread_id=thread_id,
                user_id=user_id,
                query=pending_user.strip(),
                final_answer=pending_agent.strip(),
                raw_query=pending_user.strip(),
                rewritten_query=None,
                citations=list(toolbox.citations),
            )
        pending_user = ""
        pending_agent = ""

    async def emit(event: str, data: dict[str, Any]) -> None:
        nonlocal pending_user, pending_agent

        if event == "__binary__":
            await websocket.send_bytes(data["pcm"])
            return

        if event == "user_transcript":
            pending_user += data["text"]
        elif event == "agent_transcript":
            pending_agent += data["text"]
        elif event == "interrupted":
            # The model had already said whatever is in pending_agent — the user
            # heard it before cutting in — so it's finalized as a (truncated) turn
            # here rather than left to bleed into whatever the next turn transcribes.
            await flush_pending_turn()
        elif event == "turn_complete":
            await flush_pending_turn()

        await websocket.send_json({"event": event, **data})

    async def inbound() -> dict[str, Any] | None:
        """Next browser frame: binary → mic audio, text → JSON control."""
        try:
            message = await websocket.receive()
        except (WebSocketDisconnect, RuntimeError):
            return None

        if message.get("type") == "websocket.disconnect":
            return None

        if (pcm := message.get("bytes")) is not None:
            return {"pcm": pcm}

        if (text := message.get("text")) is not None:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return {"type": "noop"}
            return parsed if isinstance(parsed, dict) else {"type": "noop"}

        return {"type": "noop"}

    await websocket.send_json({"event": "session", "thread_id": thread_id})
    logger.info("Voice session opened", user_id=user_id, thread_id=thread_id)

    session = VoiceSession(toolbox=toolbox, emit=emit, voice=requested_voice)
    try:
        await session.run(inbound)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error("Voice session error", error=str(exc), user_id=user_id)
        try:
            await websocket.close(code=WS_INTERNAL_ERROR)
        except RuntimeError:
            pass
    finally:
        logger.info("Voice session closed", user_id=user_id, thread_id=thread_id)
        try:
            await websocket.close()
        except RuntimeError:
            pass
