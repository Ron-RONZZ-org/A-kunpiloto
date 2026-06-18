"""Tests for A-kunpiloto tool executor."""

from __future__ import annotations

from unittest.mock import patch

from A_kunpiloto.tools.executor import (
    execute_tool_call,
    format_args_for_cli,
)


class TestFormatArgs:
    def test_simple_string_arg(self):
        result = format_args_for_cli({"text": "hello"})
        assert "--text" in result
        assert "hello" in result

    def test_boolean_true_uses_flag(self):
        result = format_args_for_cli({"yes": True})
        assert "--yes" in result

    def test_boolean_false_uses_no_flag(self):
        result = format_args_for_cli({"active": False})
        assert "--no-active" in result

    def test_multiple_args(self):
        result = format_args_for_cli({"name": "foo", "count": 3, "flag": True})
        # Should have flag, value, flag, value, flag
        assert "--name" in result
        assert "foo" in result
        assert "--count" in result
        assert "3" in result
        assert "--flag" in result

    def test_underscores_become_hyphens(self):
        result = format_args_for_cli({"my_param": "val"})
        assert "--my-param" in result


class TestExecuteToolCall:
    def test_unknown_tool_entry_returns_error(self, registry):
        """Executing a tool that points to a non-existent module should error."""
        from A_kunpiloto.tools.registry import ToolEntry

        fake_entry = ToolEntry(
            name="nonexistent_ls",
            module_name="nonexistent",
            display_path="nonexistent ls",
            description="Does not exist",
            is_write=False,
        )
        result = execute_tool_call(fake_entry, {})
        assert result["exit_code"] != 0

    def test_read_tool_execution(self, registry, mock_entry_points):
        """Executing a read tool like 'ls' should succeed."""
        entry = registry.get_entry("testmod_ls")
        assert entry is not None
        with patch("importlib.metadata.entry_points", return_value=mock_entry_points):
            result = execute_tool_call(entry, {"kategoria": "test"})
        assert result["exit_code"] == 0
        assert "Listing" in result["output"]

    def test_write_tool_execution(self, registry, mock_entry_points):
        """Executing a write tool like 'aldoni' should succeed."""
        entry = registry.get_entry("testmod_aldoni")
        assert entry is not None
        assert entry.positional_params == ["nomo"]
        with patch("importlib.metadata.entry_points", return_value=mock_entry_points):
            result = execute_tool_call(entry, {"nomo": "test-item", "kategoria": "general"})
        assert result["exit_code"] == 0
        assert "Added" in result["output"]

    def test_tool_with_required_arg_only(self, registry, mock_entry_points):
        """A tool with only required args works."""
        entry = registry.get_entry("testmod_aldoni")
        assert entry is not None
        with patch("importlib.metadata.entry_points", return_value=mock_entry_points):
            result = execute_tool_call(entry, {"nomo": "minimal"})
        assert result["exit_code"] == 0
