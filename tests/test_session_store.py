"""Tests for A-kunpiloto session persistence (session_store.py).

Uses tmp_path to isolate the sessions JSONL file from the real config.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from A_kunpiloto.session_store import (
    _read_all,
    edit_text,
    generate_session_id,
    list_sessions,
    load_session,
    write_assistant_message,
    write_session_start,
    write_user_message,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_sessions(tmp_path) -> Path:
    """Redirect sessions path to a temp file for test isolation."""
    session_file = tmp_path / "sessions.jsonl"
    with patch(
        "A_kunpiloto.session_store._sessions_path",
        return_value=session_file,
    ):
        yield session_file


# ---------------------------------------------------------------------------
# Session ID generation
# ---------------------------------------------------------------------------


class TestGenerateSessionId:
    def test_has_correct_format(self):
        sid = generate_session_id()
        # Format: YYYYMMDD_HHMMSS_XXXX
        parts = sid.split("_")
        assert len(parts) == 3
        assert len(parts[0]) == 8  # date
        assert len(parts[1]) == 6  # time
        assert len(parts[2]) == 4  # hex suffix
        assert parts[0].isdigit()
        assert parts[1].isdigit()

    def test_unique_per_call(self):
        ids = {generate_session_id() for _ in range(100)}
        assert len(ids) == 100  # All unique

    def test_sortable_chronologically(self):
        ids = [generate_session_id() for _ in range(5)]
        assert ids == sorted(ids)  # Timestamps increase


# ---------------------------------------------------------------------------
# Writing and reading
# ---------------------------------------------------------------------------


class TestWriteAndRead:
    def test_write_session_start(self, isolate_sessions):
        write_session_start("sess_001", {"model": "gpt-4"})
        entries = _read_all()
        assert len(entries) == 1
        assert entries[0]["type"] == "session_start"
        assert entries[0]["session_id"] == "sess_001"
        assert entries[0]["metadata"]["model"] == "gpt-4"

    def test_write_session_start_no_meta(self, isolate_sessions):
        write_session_start("sess_002")
        entries = _read_all()
        assert len(entries) == 1
        assert "metadata" not in entries[0]

    def test_write_user_message(self, isolate_sessions):
        write_session_start("sess_003")
        write_user_message("sess_003", "Hello, world!")
        entries = _read_all()
        user_msgs = [e for e in entries if e.get("role") == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "Hello, world!"

    def test_write_assistant_message(self, isolate_sessions):
        write_session_start("sess_004")
        write_assistant_message("sess_004", "Hi there!")
        entries = _read_all()
        assistant_msgs = [e for e in entries if e.get("role") == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"] == "Hi there!"

    def test_multiple_sessions_interleaved(self, isolate_sessions):
        """Entries from different sessions can coexist."""
        write_session_start("sess_a")
        write_user_message("sess_a", "msg1")
        write_session_start("sess_b")
        write_user_message("sess_b", "msg_b1")
        write_user_message("sess_a", "msg2")

        entries = _read_all()
        assert len(entries) == 5

    def test_append_to_existing(self, isolate_sessions):
        """Writing again appends, doesn't overwrite."""
        write_session_start("sess_c")
        write_user_message("sess_c", "first")
        write_user_message("sess_c", "second")

        entries = _read_all()
        user_msgs = [e for e in entries if e.get("role") == "user"]
        assert len(user_msgs) == 2

    def test_empty_file_returns_empty(self, isolate_sessions):
        entries = _read_all()
        assert entries == []

    def test_nonexistent_file_returns_empty(self):
        # When the path doesn't exist, _read_all returns []
        with patch(
            "A_kunpiloto.session_store._sessions_path",
            return_value=Path("/nonexistent/sessions.jsonl"),
        ):
            entries = _read_all()
            assert entries == []


# ---------------------------------------------------------------------------
# List sessions
# ---------------------------------------------------------------------------


class TestListSessions:
    def test_list_single_session(self, isolate_sessions):
        write_session_start("sess_001", {"model": "gpt-4"})
        write_user_message("sess_001", "Hello there")
        sessions = list_sessions(limit=10)
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "sess_001"
        assert sessions[0]["message_count"] == 1
        assert sessions[0]["model"] == "gpt-4"

    def test_list_multiple_sessions(self, isolate_sessions):
        write_session_start("sess_a")
        write_user_message("sess_a", "First session")
        write_session_start("sess_b")
        write_user_message("sess_b", "Second session")

        sessions = list_sessions(limit=10)
        assert len(sessions) == 2

    def test_list_ordered_most_recent_first(self, isolate_sessions):
        write_session_start("sess_old")
        write_user_message("sess_old", "Old session")
        write_session_start("sess_new")
        write_user_message("sess_new", "New session")

        sessions = list_sessions(limit=10)
        assert sessions[0]["session_id"] == "sess_new"
        assert sessions[1]["session_id"] == "sess_old"

    def test_list_first_message_preview(self, isolate_sessions):
        write_session_start("sess_p")
        write_user_message("sess_p", "What is the weather today?")
        sessions = list_sessions(limit=10)
        assert sessions[0]["first_message"] == "What is the weather today?"

    def test_list_first_message_truncated(self, isolate_sessions):
        long_msg = "A" * 100
        write_session_start("sess_long")
        write_user_message("sess_long", long_msg)
        sessions = list_sessions(limit=10)
        preview = sessions[0]["first_message"]
        assert len(preview) <= 83  # 80 chars + "..."
        assert preview.endswith("...")

    def test_list_empty(self, isolate_sessions):
        sessions = list_sessions(limit=10)
        assert sessions == []

    def test_list_limit(self, isolate_sessions):
        for i in range(5):
            sid = f"sess_{i}"
            write_session_start(sid)
            write_user_message(sid, f"Session {i}")
        sessions = list_sessions(limit=3)
        assert len(sessions) == 3


# ---------------------------------------------------------------------------
# Load session
# ---------------------------------------------------------------------------


class TestLoadSession:
    def test_load_returns_messages_in_order(self, isolate_sessions):
        write_session_start("sess")
        write_user_message("sess", "User message 1")
        write_assistant_message("sess", "Assistant reply 1")
        write_user_message("sess", "User message 2")

        messages = load_session("sess")
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "User message 1"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Assistant reply 1"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "User message 2"

    def test_load_only_requested_session(self, isolate_sessions):
        write_session_start("sess_a")
        write_user_message("sess_a", "Session A")
        write_session_start("sess_b")
        write_user_message("sess_b", "Session B")

        messages = load_session("sess_a")
        assert len(messages) == 1
        assert messages[0]["content"] == "Session A"

    def test_load_empty_session(self, isolate_sessions):
        write_session_start("sess_empty")
        messages = load_session("sess_empty")
        assert messages == []

    def test_load_nonexistent(self, isolate_sessions):
        messages = load_session("does_not_exist")
        assert messages == []

    def test_load_skips_non_message_entries(self, isolate_sessions):
        write_session_start("sess")
        write_user_message("sess", "Hello")
        write_assistant_message("sess", "Hi")
        messages = load_session("sess")
        for m in messages:
            assert m["role"] in ("user", "assistant")
            assert "content" in m


# ---------------------------------------------------------------------------
# Edit text (interactive editing via $EDITOR)
# ---------------------------------------------------------------------------


class TestEditText:
    def test_no_editor_returns_none(self):
        with patch("os.environ.get", return_value=""):
            result = edit_text("test")
            assert result is None

    def test_editor_returns_content(self):
        """With a simple 'cat' editor, content should pass through."""
        with patch("os.environ.get", return_value="cat"):
            with patch("tempfile.NamedTemporaryFile") as mock_tmp:
                mock_file = mock_tmp.return_value.__enter__.return_value
                mock_file.name = "/tmp/test_edit.txt"

                with patch("os.system"):
                    with patch("builtins.open") as mock_open:
                        mock_fh = mock_open.return_value.__enter__.return_value
                        mock_fh.read.return_value = "edited content"

                        result = edit_text("original")
                        assert result == "edited content"

    def test_editor_failure_returns_none(self):
        with patch("os.environ.get", return_value="false"):
            with patch("tempfile.NamedTemporaryFile") as mock_tmp:
                mock_file = mock_tmp.return_value.__enter__.return_value
                mock_file.name = "/tmp/test_edit.txt"

                with patch("os.system"):
                    with patch("builtins.open", side_effect=OSError("read error")):
                        result = edit_text("original")
                        assert result is None
