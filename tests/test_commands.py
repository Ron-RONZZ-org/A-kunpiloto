"""Tests for A-kunpiloto custom commands parsing and template resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from A_kunpiloto.commands import (
    CommandDef,
    _parse_frontmatter,
    _strip_frontmatter,
    find_command,
    load_commands,
    resolve_template,
)


# ---------------------------------------------------------------------------
# Frontmatter parser
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_no_frontmatter(self):
        result = _parse_frontmatter("just text")
        assert result == {}

    def test_empty_frontmatter(self):
        text = "---\n---\nbody"
        result = _parse_frontmatter(text)
        assert result == {}

    def test_simple_description(self):
        text = "---\ndescription: Show today's events\n---\nbody"
        result = _parse_frontmatter(text)
        assert result == {"description": "Show today's events"}

    def test_multiple_fields(self):
        text = (
            "---\n"
            "description: Show dashboard\n"
            "model: gpt-4\n"
            "---\n"
            "body"
        )
        result = _parse_frontmatter(text)
        assert result == {"description": "Show dashboard", "model": "gpt-4"}

    def test_quoted_value(self):
        text = '---\ndescription: "My Command"\n---\nbody'
        result = _parse_frontmatter(text)
        assert result == {"description": "My Command"}

    def test_single_quoted_value(self):
        text = "---\ndescription: 'My Command'\n---\nbody"
        result = _parse_frontmatter(text)
        assert result == {"description": "My Command"}

    def test_comment_lines_ignored(self):
        text = (
            "---\n"
            "description: Test\n"
            "# this is a comment\n"
            "---\n"
            "body"
        )
        result = _parse_frontmatter(text)
        assert result == {"description": "Test"}

    def test_trailing_spaces_in_value(self):
        text = "---\ndescription:   My Command   \n---\nbody"
        result = _parse_frontmatter(text)
        assert result == {"description": "My Command"}

    def test_multiline_body_preserved(self):
        text = "---\ndescription: test\n---\nLine 1\n\nLine 2"
        result = _parse_frontmatter(text)
        assert result == {"description": "test"}


# ---------------------------------------------------------------------------
# Frontmatter stripping
# ---------------------------------------------------------------------------


class TestStripFrontmatter:
    def test_no_frontmatter(self):
        assert _strip_frontmatter("hello world") == "hello world"

    def test_with_frontmatter(self):
        text = "---\ndescription: test\n---\n  body text  "
        assert _strip_frontmatter(text) == "body text"

    def test_only_frontmatter(self):
        text = "---\ndescription: test\n---\n  "
        assert _strip_frontmatter(text) == ""

    def test_multiline_body(self):
        text = "---\ndescription: test\n---\nLine 1\nLine 2\n"
        assert _strip_frontmatter(text) == "Line 1\nLine 2"


# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------


class TestResolveTemplate:
    def test_no_placeholders(self):
        assert resolve_template("Hello world", []) == "Hello world"

    def test_arguments_placeholder(self):
        result = resolve_template("Search for $ARGUMENTS", ["python", "testing"])
        assert result == "Search for python testing"

    def test_at_symbol(self):
        result = resolve_template("Search for $@", ["python", "testing"])
        assert result == "Search for python testing"

    def test_positional_args(self):
        result = resolve_template("Create $1 with $2", ["file.txt", "content"])
        assert result == "Create file.txt with content"

    def test_positional_out_of_range(self):
        """$3 should be left as-is when only 2 args provided."""
        result = resolve_template("$1 and $2 and $3", ["a", "b"])
        assert result == "a and b and $3"

    def test_no_args_all_placeholders_remain(self):
        result = resolve_template("$1 and $ARGUMENTS", [])
        assert result == "$1 and "

    def test_mixed_placeholders(self):
        result = resolve_template(
            "$1: $ARGUMENTS end",
            ["cmd", "arg1", "arg2"],
        )
        assert result == "cmd: cmd arg1 arg2 end"

    def test_literal_dollar(self):
        result = resolve_template("Price: $$10", [])
        assert result == "Price: $10"

    def test_literal_dollar_with_args(self):
        result = resolve_template("$$ARGUMENTS", ["test"])
        assert result == "$ARGUMENTS"

    def test_dollar_0_left_as_is(self):
        result = resolve_template("$0 command", ["arg"])
        assert result == "$0 command"


# ---------------------------------------------------------------------------
# Loading commands from disk
# ---------------------------------------------------------------------------


class TestLoadCommands:
    def test_nonexistent_dir_returns_empty(self, tmp_path):
        result = load_commands(tmp_path / "nonexistent")
        assert result == []

    def test_empty_dir_returns_empty(self, tmp_path):
        cmd_dir = tmp_path / "commands"
        cmd_dir.mkdir(parents=True)
        result = load_commands(cmd_dir)
        assert result == []

    def test_single_command(self, tmp_path):
        cmd_dir = tmp_path / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "today.md").write_text(
            "---\ndescription: Today's summary\n---\n"
            "What emails today?",
            encoding="utf-8",
        )
        result = load_commands(cmd_dir)
        assert len(result) == 1
        assert result[0].name == "today"
        assert result[0].description == "Today's summary"
        assert result[0].template == "What emails today?"

    def test_multiple_commands(self, tmp_path):
        cmd_dir = tmp_path / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "today.md").write_text(
            "---\ndescription: Today\n---\nEmails?",
            encoding="utf-8",
        )
        (cmd_dir / "weekend.md").write_text(
            "---\ndescription: Weekend\n---\nWeekend plan?",
            encoding="utf-8",
        )
        result = load_commands(cmd_dir)
        assert len(result) == 2
        names = [c.name for c in result]
        assert "today" in names
        assert "weekend" in names

    def test_commands_sorted_alphabetically(self, tmp_path):
        cmd_dir = tmp_path / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "z.md").write_text("Z command", encoding="utf-8")
        (cmd_dir / "a.md").write_text("A command", encoding="utf-8")
        (cmd_dir / "m.md").write_text("M command", encoding="utf-8")
        result = load_commands(cmd_dir)
        assert [c.name for c in result] == ["a", "m", "z"]

    def test_ignores_non_md_files(self, tmp_path):
        cmd_dir = tmp_path / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "today.md").write_text("valid", encoding="utf-8")
        (cmd_dir / "notes.txt").write_text("ignored", encoding="utf-8")
        (cmd_dir / "script.py").write_text("ignored", encoding="utf-8")
        result = load_commands(cmd_dir)
        assert len(result) == 1
        assert result[0].name == "today"

    def test_skips_empty_body(self, tmp_path):
        cmd_dir = tmp_path / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "empty.md").write_text("---\ndescription: empty\n---\n   \n", encoding="utf-8")
        (cmd_dir / "real.md").write_text("Real command", encoding="utf-8")
        result = load_commands(cmd_dir)
        assert len(result) == 1
        assert result[0].name == "real"

    def test_handles_os_error_gracefully(self, tmp_path):
        cmd_dir = tmp_path / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "good.md").write_text("Good command", encoding="utf-8")
        bad = cmd_dir / "bad.md"
        bad.write_text("Bad", encoding="utf-8")
        bad.chmod(0o000)  # Make unreadable
        result = load_commands(cmd_dir)
        # Good command should still be loaded
        assert len(result) == 1
        assert result[0].name == "good"
        bad.chmod(0o644)  # Restore permissions

    def test_no_frontmatter(self, tmp_path):
        cmd_dir = tmp_path / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "simple.md").write_text(
            "Just a simple template without frontmatter",
            encoding="utf-8",
        )
        result = load_commands(cmd_dir)
        assert len(result) == 1
        assert result[0].name == "simple"
        assert result[0].description == ""
        assert result[0].template == "Just a simple template without frontmatter"


# ---------------------------------------------------------------------------
# find_command
# ---------------------------------------------------------------------------


class TestFindCommand:
    def setup_method(self):
        self.commands = [
            CommandDef(name="today", description="Today", template="Today?"),
            CommandDef(name="WeekEnd", description="Weekend", template="Weekend?"),
        ]

    def test_find_exact(self):
        cmd = find_command(self.commands, "today")
        assert cmd is not None
        assert cmd.name == "today"

    def test_find_case_insensitive(self):
        cmd = find_command(self.commands, "TODAY")
        assert cmd is not None
        assert cmd.name == "today"

    def test_find_case_insensitive_source(self):
        cmd = find_command(self.commands, "weekend")
        assert cmd is not None
        assert cmd.name == "WeekEnd"

    def test_not_found(self):
        cmd = find_command(self.commands, "nonexistent")
        assert cmd is None

    def test_empty_list(self):
        cmd = find_command([], "today")
        assert cmd is None
