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
