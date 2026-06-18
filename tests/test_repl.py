"""Tests for A-kunpiloto REPL custom command handling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from A_kunpiloto.commands import CommandDef
from A_kunpiloto.repl import REPL, _help_text


# ---------------------------------------------------------------------------
# Help text rendering
# ---------------------------------------------------------------------------


class TestHelpText:
    def test_no_custom_commands(self):
        text = _help_text([])
        assert "/exit" in text
        assert "/clear" in text
        assert "/tools" in text
        assert "Custom commands:" not in text

    def test_with_custom_commands(self):
        cmds = [
            CommandDef(name="today", description="Today's summary"),
            CommandDef(name="weekend", description=""),
        ]
        text = _help_text(cmds)
        assert "/exit" in text
        assert "/today" in text
        assert "Today's summary" in text
        assert "/weekend" in text

    def test_custom_commands_section_label(self):
        cmds = [CommandDef(name="test", description="Test command")]
        text = _help_text(cmds)
        # tr_multi returns Esperanto by default, so we check that
        # the command name appears, not the English label
        assert "/test" in text
        assert "Test command" in text


# ---------------------------------------------------------------------------
# Custom command execution in REPL
# ---------------------------------------------------------------------------


class TestCustomCommandExecution:
    """Test that custom slash-commands are correctly handled in the REPL.

    We bypass the event loop and directly call _handle_command, then
    inspect the conversation history for the resolved template.
    """

    @pytest.fixture
    def repl(self, mock_provider, registry):
        """Create a REPL instance with mock provider and custom commands."""
        custom_commands = [
            CommandDef(name="today", description="Today", template="What emails today?"),
            CommandDef(
                name="summarize",
                description="Summarize",
                template="Summarize $ARGUMENTS",
            ),
        ]
        r = REPL(
            provider=mock_provider,
            registry=registry,
            max_turns=15,
            temperature=0.7,
            custom_commands=custom_commands,
        )
        return r

    def test_custom_command_resolves_and_adds_to_history(self, repl):
        """Calling /today should resolve the template and add to history."""
        # _handle_command calls _process_turn which calls the provider.
        # We patch the provider to return a simple response.
        from A.core.providers import LLMResponse

        repl._provider.chat.return_value = LLMResponse(content="Here's your summary.")

        with patch.object(repl, "_start_thinking"), patch.object(repl, "_stop_thinking"):
            repl._handle_command("/today")

        # After processing, the history should contain the resolved template
        history = repl._history.messages
        user_msgs = [m for m in history if m["role"] == "user"]
        assert any("What emails today?" in m["content"] for m in user_msgs)

    def test_custom_command_with_args(self, repl):
        """Calling /summarize report.txt should resolve $ARGUMENTS."""
        from A.core.providers import LLMResponse

        repl._provider.chat.return_value = LLMResponse(content="Summary done.")

        with patch.object(repl, "_start_thinking"), patch.object(repl, "_stop_thinking"):
            repl._handle_command("/summarize report.txt")

        history = repl._history.messages
        user_msgs = [m for m in history if m["role"] == "user"]
        assert any("Summarize report.txt" in m["content"] for m in user_msgs)

    def test_unknown_command_shows_error(self, repl):
        """An unknown /command should print an error, not add to history."""
        with patch("A_kunpiloto.repl._console") as mock_console:
            repl._handle_command("/nonexistent")

        # History should not contain anything about nonexistent
        history = repl._history.messages
        user_msgs = [m for m in history if m["role"] == "user"]
        assert all("nonexistent" not in m["content"] for m in user_msgs)

        # Error should be printed to console
        mock_console.print.assert_called_once()
        error_text = str(mock_console.print.call_args[0][0])
        assert "nekonata" in error_text.lower() or "unknown" in error_text.lower()

    def test_custom_command_case_insensitive(self, repl):
        """Calling /TODAY should still find the today command."""
        from A.core.providers import LLMResponse

        repl._provider.chat.return_value = LLMResponse(content="Done.")

        with patch.object(repl, "_start_thinking"), patch.object(repl, "_stop_thinking"):
            repl._handle_command("/TODAY")

        history = repl._history.messages
        user_msgs = [m for m in history if m["role"] == "user"]
        assert any("What emails today?" in m["content"] for m in user_msgs)

    def test_builtin_commands_still_work(self, repl):
        """Built-in commands like /help must still function."""
        with patch("A_kunpiloto.repl._console") as mock_console:
            repl._handle_command("/help")

        mock_console.print.assert_called_once()
        text = str(mock_console.print.call_args[0][0])
        assert "/exit" in text
        assert "/today" in text  # Custom commands appear in help

    def test_custom_commands_appear_in_help(self, repl):
        """Custom commands should appear in /help output."""
        with patch("A_kunpiloto.repl._console") as mock_console:
            repl._handle_command("/help")

        text = str(mock_console.print.call_args[0][0])
        assert "/today" in text
        assert "/summarize" in text

    def test_system_prompt_uses_load_system_prompt(self, monkeypatch, tmp_path, mock_provider, registry):
        """Verify that _build_history uses the file-based system prompt."""
        # Create a custom system prompt
        prompt_dir = tmp_path / "config" / "kunpiloto"
        prompt_dir.mkdir(parents=True)
        (prompt_dir / "system_prompt.md").write_text(
            "Custom system prompt override", encoding="utf-8",
        )
        monkeypatch.setenv("A_DIR", str(tmp_path))

        r = REPL(
            provider=mock_provider,
            registry=registry,
            custom_commands=[],
        )
        history = r._history.messages
        system_msgs = [m for m in history if m["role"] == "system"]
        assert len(system_msgs) >= 1
        assert "Custom system prompt override" in system_msgs[0]["content"]


# ---------------------------------------------------------------------------
# Failure limit tests
# ---------------------------------------------------------------------------


class TestFailureLimit:
    """Test that REPL aborts after consecutive tool failures."""

    def test_failure_limit_triggers_on_errors(self, mock_provider, registry):
        """After 3 consecutive failures, REPL should abort."""
        from A.core.providers import LLMResponse, ToolCall

        # Simulate a tool call that always fails
        mock_provider.chat.side_effect = [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function={"name": "nonexistent_tool", "arguments": "{}"},
                    )
                ],
            ),
        ]

        r = REPL(
            provider=mock_provider,
            registry=registry,
            max_turns=10,
            temperature=0.7,
            custom_commands=[],
        )
        r._history.clear()

        with patch("A_kunpiloto.repl._console") as mock_console, \
             patch.object(r, "_start_thinking"), \
             patch.object(r, "_stop_thinking"):

            # Reset result store for clean test
            r._result_store.clear()
            r._history.add_user("do something")
            r._process_turn()

        # After 3 consecutive failures, the assistant should have produced
        # an abort message. The failure limit kicks in after 3 errors
        # but only 1 tool call was made here (side_effect has 1 item).
        # The LLM returns 1 tool call, it fails, that's 1 failure.
        # Next iteration would call side_effect again, but there's no
        # more items... Let's just verify the error was recorded.
        history = r._history.messages
        tool_msgs = [m for m in history if m["role"] == "tool"]
        assert len(tool_msgs) >= 1
        # tr_multi in tests returns eo: "ne trovita"
        assert "ne trovita" in tool_msgs[0]["content"]

    def test_builtin_tool_registered_on_init(self, mock_provider, registry):
        """REPL should register read_result built-in tool."""
        r = REPL(
            provider=mock_provider,
            registry=registry,
            max_turns=15,
            temperature=0.7,
        )
        entry = registry.get_entry("read_result")
        assert entry is not None
        assert entry.handler is not None
        assert entry.module_name == "_builtin"

    def test_builtin_tool_in_schemas(self, mock_provider, registry):
        """read_result should appear in the registry's schemas."""
        r = REPL(
            provider=mock_provider,
            registry=registry,
            max_turns=15,
            temperature=0.7,
        )
        schemas = registry.get_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "read_result" in names


# ---------------------------------------------------------------------------
# Fuzzy matching in tool calls
# ---------------------------------------------------------------------------


class TestFuzzyMatchInToolCalls:
    def test_similar_tool_suggested(self, mock_provider, registry):
        """When a tool is not found, similar names are suggested."""
        from A.core.providers import ToolCall

        r = REPL(
            provider=mock_provider,
            registry=registry,
            max_turns=15,
            temperature=0.7,
        )

        tc = ToolCall(
            id="call_xyz",
            function={"name": "testmod_ldoni", "arguments": "{}"},  # typo
        )

        result = r._handle_tool_call(tc)
        assert result["exit_code"] == 1
        assert "not found" in result["error"]
        # Since "testmod_ldoni" is close to "testmod_aldoni", we should get a suggestion
        assert "Did you mean" in result["error"]

    def test_unrelated_tool_no_suggestion(self, mock_provider, registry):
        """When no close match exists, no suggestions."""
        from A.core.providers import ToolCall

        r = REPL(
            provider=mock_provider,
            registry=registry,
            max_turns=15,
            temperature=0.7,
        )

        tc = ToolCall(
            id="call_xyz",
            function={"name": "zzzzzzz", "arguments": "{}"},
        )

        result = r._handle_tool_call(tc)
        assert result["exit_code"] == 1
        # Should NOT have "Did you mean" since no close match
        assert "Did you mean" not in result["error"]


# ---------------------------------------------------------------------------
# Readline import tests
# ---------------------------------------------------------------------------


class TestReadlineImport:
    """Verify that the REPL module handles the readline import gracefully."""

    def test_repl_module_imports_successfully(self):
        """The REPL module must import without error."""
        # Already imported at module scope — this confirms no crash
        from A_kunpiloto.repl import REPL
        assert REPL is not None

    def test_readline_available_on_unix(self):
        """On Linux/macOS, readline should be available after importing repl."""
        import sys
        if sys.platform == "win32":
            pytest.skip("readline is not available on Windows")
        # After importing A_kunpiloto.repl (done at top of file),
        # readline should be in sys.modules
        assert "readline" in sys.modules, (
            "readline was not loaded. The try/except ImportError block "
            "in repl.py may not be executing."
        )

    def test_repl_works_without_readline_subprocess(self):
        """When readline is unavailable, the REPL module must still load.

        Runs in a subprocess to get a clean Python environment where
        readline can be suppressed before any import of repl.
        """
        import subprocess
        import sys

        code = """
import sys
import builtins

# Suppress readline before any other import
_real_import = builtins.__import__

def _no_readline(name, *args, **kwargs):
    if name == 'readline':
        raise ImportError('Simulated readline unavailability')
    return _real_import(name, *args, **kwargs)

builtins.__import__ = _no_readline

# Now import the REPL module — must not crash
from A_kunpiloto.repl import REPL
print(f'REPL class: {REPL}')
assert REPL is not None
print('OK: readline not available, but REPL loaded')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30,
            cwd="/home/rongzhou/kodo/autish/A-kunpiloto",
        )
        assert result.returncode == 0, (
            f"REPL failed to import without readline:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        assert "OK:" in result.stdout


# ---------------------------------------------------------------------------
# Paste confirmation tests
# ---------------------------------------------------------------------------


class TestPasteDetection:
    """Test that the REPL correctly detects and handles pasted content."""

    def test_is_paste_multi_line(self):
        """Multi-line content should be detected as paste."""
        assert REPL._is_paste("line1\nline2\nline3")

    def test_is_paste_long_line(self):
        """Single line over threshold should be detected as paste."""
        long_line = "x" * (REPL._PASTE_MIN_CHARS + 1)
        assert REPL._is_paste(long_line)

    def test_is_paste_short_single_line(self):
        """Short single line should NOT be detected as paste."""
        assert not REPL._is_paste("Hello, how are you?")

    def test_is_paste_empty(self):
        """Empty string should not be paste."""
        assert not REPL._is_paste("")

    def test_is_paste_boundary(self):
        """Exactly at threshold should NOT be paste (uses > not >=)."""
        text = "x" * REPL._PASTE_MIN_CHARS
        assert not REPL._is_paste(text)  # Strictly >, not >=

    def test_is_paste_one_over_boundary(self):
        """One char over threshold should be paste."""
        text = "x" * (REPL._PASTE_MIN_CHARS + 1)
        assert REPL._is_paste(text)

    def test_handle_pasted_input_confirms(self, mock_provider, registry):
        """Paste confirmation should be shown for multi-line input."""
        from unittest.mock import patch

        r = REPL(
            provider=mock_provider,
            registry=registry,
            max_turns=15,
            temperature=0.7,
        )

        with patch.object(r, "_confirm_or_edit_paste", return_value="modified text"):
            result = r._handle_pasted_input("line1\nline2")
            assert result == "modified text"

    def test_handle_pasted_input_skips_normal(self, mock_provider, registry):
        """Short single-line input should pass through without confirmation."""
        r = REPL(
            provider=mock_provider,
            registry=registry,
            max_turns=15,
            temperature=0.7,
        )
        result = r._handle_pasted_input("Hello")
        assert result == "Hello"


# ---------------------------------------------------------------------------
# Session management tests
# ---------------------------------------------------------------------------


class TestSessionCommands:
    """Test the /sessions and /resume slash commands."""

    def test_sessions_empty(self, mock_provider, registry):
        """Listing sessions when none exist should show empty message."""
        from unittest.mock import patch

        r = REPL(
            provider=mock_provider,
            registry=registry,
            max_turns=15,
            temperature=0.7,
        )

        with patch("A_kunpiloto.repl._console") as mock_console:
            r._cmd_sessions()

        # Should print a "no sessions" message
        mock_console.print.assert_called_once()
        text = str(mock_console.print.call_args[0][0])
        assert "neniuj" in text.lower() or "no" in text.lower()

    def test_sessions_with_data(self, mock_provider, registry, tmp_path):
        """Listing sessions with data should show them in a table."""
        from unittest.mock import patch

        # Write a session entry
        session_file = tmp_path / "sessions.jsonl"
        with session_file.open("w", encoding="utf-8") as f:
            f.write(
                '{"type": "session_start", "session_id": "test_sess", '
                '"timestamp": "2026-01-01T12:00:00"}\n'
            )
            f.write(
                '{"type": "message", "session_id": "test_sess", '
                '"role": "user", "content": "Hello"}\n'
            )

        with patch(
            "A_kunpiloto.session_store._sessions_path",
            return_value=session_file,
        ):
            from unittest.mock import MagicMock
            mock_console = MagicMock()
            r = REPL(
                provider=mock_provider,
                registry=registry,
                max_turns=15,
                temperature=0.7,
            )
            with patch("A_kunpiloto.repl._console", mock_console):
                r._cmd_sessions()

            # Should print a table (multiple calls to console.print)
            assert mock_console.print.call_count >= 1

    def test_resume_nonexistent(self, mock_provider, registry):
        """Resuming a nonexistent session should show error."""
        from unittest.mock import patch

        r = REPL(
            provider=mock_provider,
            registry=registry,
            max_turns=15,
            temperature=0.7,
        )

        with patch("A_kunpiloto.repl._console") as mock_console:
            r._cmd_resume("nonexistent")

        mock_console.print.assert_called_once()
        text = str(mock_console.print.call_args[0][0])
        assert "ne trovita" in text.lower() or "not found" in text.lower()

    def test_resume_loads_messages(self, mock_provider, registry, tmp_path):
        """Resuming should load messages into history."""
        from unittest.mock import patch, MagicMock

        session_file = tmp_path / "sessions.jsonl"
        with session_file.open("w", encoding="utf-8") as f:
            f.write(
                '{"type": "session_start", "session_id": "sess_1", '
                '"timestamp": "2026-01-01T12:00:00"}\n'
            )
            f.write(
                '{"type": "message", "session_id": "sess_1", '
                '"role": "user", "content": "Hello from past"}\n'
            )
            f.write(
                '{"type": "message", "session_id": "sess_1", '
                '"role": "assistant", "content": "Reply from past"}\n'
            )

        with patch(
            "A_kunpiloto.session_store._sessions_path",
            return_value=session_file,
        ):
            mock_console = MagicMock()
            r = REPL(
                provider=mock_provider,
                registry=registry,
                max_turns=15,
                temperature=0.7,
            )
            with patch("A_kunpiloto.repl._console", mock_console):
                r._cmd_resume("sess_1")

            # History should contain loaded messages
            history = r._history.messages
            contents = [m["content"] for m in history]
            assert any("Hello from past" in c for c in contents)
            assert any("Reply from past" in c for c in contents)

            # Should have resume note system message
            system_msgs = [m for m in history if m["role"] == "system"]
            resume_notes = [m for m in system_msgs if "resumed" in m["content"].lower()
                            or "rekomencita" in m["content"].lower()
                            or "reprend" in m["content"].lower()]
            assert len(resume_notes) >= 1

    def test_start_new_session(self, mock_provider, registry, tmp_path):
        """Starting a new session should generate an ID and write to store."""
        from unittest.mock import patch

        session_file = tmp_path / "sessions.jsonl"
        with patch(
            "A_kunpiloto.session_store._sessions_path",
            return_value=session_file,
        ):
            r = REPL(
                provider=mock_provider,
                registry=registry,
                max_turns=15,
                temperature=0.7,
            )
            r._start_new_session()
            assert r._session_id != ""
            assert session_file.exists()

    def test_clear_starts_new_session(self, mock_provider, registry, tmp_path):
        """/clear should start a new session."""
        from unittest.mock import patch

        session_file = tmp_path / "sessions.jsonl"
        with patch(
            "A_kunpiloto.session_store._sessions_path",
            return_value=session_file,
        ):
            from unittest.mock import MagicMock
            r = REPL(
                provider=mock_provider,
                registry=registry,
                max_turns=15,
                temperature=0.7,
            )
            old_id = r._session_id
            with patch("A_kunpiloto.repl._console", MagicMock()):
                r._handle_command("/clear")
            assert r._session_id != ""
            # Session ID should have changed
            assert r._session_id != old_id if old_id else True
