"""Session persistence for A-kunpiloto — saves/loads conversation history.

Uses an append-only JSONL file at ``~/.config/A/kunpiloto/sessions.jsonl``.
Each line is one JSON object representing a session event (start, user
message, assistant response).

Only **user** and **assistant (text-only)** messages are persisted.
Tool calls and tool results are NOT stored — they contain live data
that would be stale on resume.

On resume, a system note is injected explaining that previous tool
results are unavailable.
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from A import tr_multi
from A_kunpiloto.config import config_path

_SESSIONS_FILENAME = "sessions.jsonl"
_MAX_SESSION_LINES = 50_000  # Safety limit to prevent unbounded file growth


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _sessions_path() -> Path:
    """Return the path to the sessions JSONL file."""
    # Sessions live next to the config file: ~/.config/A/kunpiloto/
    return config_path().parent / _SESSIONS_FILENAME


# ---------------------------------------------------------------------------
# Session ID generation
# ---------------------------------------------------------------------------


def generate_session_id() -> str:
    """Generate a unique, human-readable session ID.

    Format: ``YYYYMMDD_HHMMSS_XXXX`` where XXXX is a 4-char hex
    random suffix for disambiguation.

    Returns:
        A session ID string.
    """
    now = datetime.now()
    suffix = secrets.token_hex(2)  # 4 hex chars
    return now.strftime("%Y%m%d_%H%M%S") + f"_{suffix}"


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _append_line(data: dict[str, Any]) -> None:
    """Append one JSON line to the sessions file, creating dirs if needed.

    Args:
        data: The dict to serialize as JSON.
    """
    path = _sessions_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
    # Trim file if it exceeds the safety limit
    _trim_file(path)


def _trim_file(path: Path) -> None:
    """Remove oldest sessions if file exceeds MAX_SESSION_LINES lines."""
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) > _MAX_SESSION_LINES:
            # Keep the most recent MAX_SESSION_LINES lines
            path.write_text(
                "\n".join(lines[-_MAX_SESSION_LINES:]) + "\n",
                encoding="utf-8",
            )
    except OSError:
        pass  # Non-critical, best-effort


def write_session_start(
    session_id: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record the start of a new session.

    Args:
        session_id: The session ID.
        metadata: Optional extra metadata (model, provider, etc.).
    """
    entry: dict[str, Any] = {
        "type": "session_start",
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
    }
    if metadata:
        entry["metadata"] = metadata
    _append_line(entry)


def write_user_message(session_id: str, content: str) -> None:
    """Record a user message.

    Args:
        session_id: The session ID.
        content: The user's message text.
    """
    _append_line({
        "type": "message",
        "session_id": session_id,
        "role": "user",
        "content": content,
        "timestamp": datetime.now().isoformat(),
    })


def write_assistant_message(session_id: str, content: str) -> None:
    """Record an assistant text-only message.

    Only pure text responses are persisted — messages that include
    ``tool_calls`` are skipped during replay.

    Args:
        session_id: The session ID.
        content: The assistant's text response.
    """
    _append_line({
        "type": "message",
        "session_id": session_id,
        "role": "assistant",
        "content": content,
        "timestamp": datetime.now().isoformat(),
    })


# ---------------------------------------------------------------------------
# Reading / Querying
# ---------------------------------------------------------------------------


def _read_all() -> list[dict[str, Any]]:
    """Read all entries from the sessions file.

    Returns:
        List of parsed JSON dicts. Malformed lines are skipped.
    """
    path = _sessions_path()
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def list_sessions(limit: int = 10) -> list[dict[str, Any]]:
    """List the most recent sessions with metadata.

    Each session is summarised by its start timestamp, the first user
    message (truncated), model info, and message count.

    Args:
        limit: Maximum number of sessions to return.

    Returns:
        A list of session summary dicts, most recent first:
        ``session_id``, ``timestamp``, ``first_message``, ``message_count``,
        ``model``.
    """
    entries = _read_all()
    sessions: dict[str, dict[str, Any]] = {}

    for entry in entries:
        sid = entry.get("session_id", "")
        if not sid:
            continue
        if sid not in sessions:
            sessions[sid] = {
                "session_id": sid,
                "timestamp": "",
                "first_message": "",
                "message_count": 0,
                "model": "",
            }
        typ = entry.get("type", "")
        if typ == "session_start":
            sessions[sid]["timestamp"] = entry.get("timestamp", "")
            meta = entry.get("metadata", {})
            if meta:
                sessions[sid]["model"] = meta.get("model", "")
        elif entry.get("role") == "user":
            content = entry.get("content", "")
            if not sessions[sid]["first_message"] and content:
                preview = content[:80].replace("\n", " ")
                if len(content) > 80:
                    preview += "..."
                sessions[sid]["first_message"] = preview
            sessions[sid]["message_count"] += 1
        elif entry.get("role") == "assistant":
            sessions[sid]["message_count"] += 1

    # Sort by timestamp descending (most recent first)
    sorted_sessions = sorted(
        sessions.values(),
        key=lambda s: s["timestamp"],
        reverse=True,
    )
    return sorted_sessions[:limit]


def load_session(session_id: str) -> list[dict[str, Any]]:
    """Load all user and assistant messages for a given session.

    Returns only ``{"role": "user", "content": "..."}`` and
    ``{"role": "assistant", "content": "..."}`` dicts in chronological
    order.  Messages that originally contained ``tool_calls`` are
    excluded — they are stale and would mislead the LLM.

    Args:
        session_id: The session ID to load.

    Returns:
        A list of message dicts suitable for inserting into
        ``ConversationHistory``.  Empty list if the session is not found.
    """
    entries = _read_all()
    messages: list[dict[str, Any]] = []

    for entry in entries:
        if entry.get("session_id") != session_id:
            continue
        if entry.get("type") != "message":
            continue
        role = entry.get("role", "")
        if role not in ("user", "assistant"):
            continue
        content = entry.get("content", "")
        if not content:
            continue
        messages.append({
            "role": role,
            "content": content,
        })

    return messages


# ---------------------------------------------------------------------------
# Interactive text editing
# ---------------------------------------------------------------------------


def edit_text(text: str) -> str | None:
    """Open the user's ``$EDITOR`` to modify a text.

    Args:
        text: The initial text to show in the editor.

    Returns:
        The edited text, or ``None`` if the user cancelled or no
        editor is available.
    """
    editor = os.environ.get("EDITOR", "")
    if not editor:
        return None

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8",
    ) as f:
        f.write(text)
        temp_path = f.name

    try:
        os.system(f"{editor} {temp_path}")
        with open(temp_path, encoding="utf-8") as f:
            edited = f.read()
        return edited
    except OSError:
        return None
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
