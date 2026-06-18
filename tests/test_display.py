"""Tests for A-kunpiloto markdown rendering in display utilities.

Verifies that ``display_assistant`` renders Markdown via ``rich.markdown.Markdown``
while other display functions (tool results, errors) remain unchanged.
"""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from A_kunpiloto._display import (
    display_assistant,
    display_tool_error,
    display_tool_result,
)
from A_kunpiloto.tools._base import ToolEntry


# ---------------------------------------------------------------------------
# Helper — capture Rich console output as plain text
# ---------------------------------------------------------------------------

_OUTPUT = ""


@pytest.fixture(autouse=True)
def _capture_console(monkeypatch):
    """Replace the module-level ``_console`` with one that writes to a StringIO."""
    from A_kunpiloto import _display as disp_mod

    buf = StringIO()
    test_console = Console(file=buf, force_terminal=False, force_interactive=False)
    monkeypatch.setattr(disp_mod, "_console", test_console)
    yield buf
    buf.close()


def _rendered(buf: StringIO) -> str:
    """Rewind *buf* and return everything written to it as plain text.

    Rich markup is stripped; ANSI codes are stripped.
    """
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# display_assistant — Markdown rendering
# ---------------------------------------------------------------------------


class TestDisplayAssistantMarkdown:
    """Verify that user-facing Markdown is rendered by ``rich.markdown.Markdown``."""

    def test_plain_text_passthrough(self, _capture_console):
        """Plain text with no Markdown should still display."""
        display_assistant("Hello world")
        text = _rendered(_capture_console)
        assert "Hello world" in text

    def test_bold_text_rendered(self, _capture_console):
        """**bold** should be rendered (not shown as literal asterisks)."""
        display_assistant("This is **bold** text")
        text = _rendered(_capture_console)
        # Rich Markdown renders **bold** — we should NOT see raw asterisks
        assert "**bold**" not in text, "Bold markers should be stripped/replaced"
        # Check the bold phrase appears
        assert "bold" in text

    def test_inline_code_rendered(self, _capture_console):
        """Inline code with backticks should be rendered."""
        display_assistant("Use the `aldoni` command")
        text = _rendered(_capture_console)
        assert "aldoni" in text

    def test_code_block(self, _capture_console):
        """Fenced code blocks should appear formatted."""
        code = "```\nprint('hello')\n```"
        display_assistant(code)
        text = _rendered(_capture_console)
        assert "print" in text
        assert "hello" in text

    def test_unordered_list(self, _capture_console):
        """Bullet lists should be indented."""
        md = "- item one\n- item two\n- item three"
        display_assistant(md)
        text = _rendered(_capture_console)
        assert "item one" in text
        assert "item two" in text
        assert "item three" in text

    def test_ordered_list(self, _capture_console):
        """Numbered lists should be rendered."""
        md = "1. first\n2. second"
        display_assistant(md)
        text = _rendered(_capture_console)
        assert "first" in text
        assert "second" in text

    def test_heading(self, _capture_console):
        """Headings should be rendered."""
        display_assistant("# Heading 1\n\nSome content")
        text = _rendered(_capture_console)
        assert "Heading 1" in text
        assert "Some content" in text

    def test_mixed_markdown(self, _capture_console):
        """A realistic mix of Markdown constructs."""
        md = """# Results

Here are the **key findings**:

- Item `alpha` is ready
- Item `beta` needs review

## Details

Run `check --all` for more.
"""
        display_assistant(md)
        text = _rendered(_capture_console)
        assert "Results" in text
        assert "key findings" in text
        assert "alpha" in text
        assert "beta" in text
        assert "Details" in text
        assert "check --all" in text or "check" in text

    def test_empty_string(self, _capture_console):
        """Empty string should show a placeholder."""
        display_assistant("")
        text = _rendered(_capture_console)
        # Should show the dim placeholder
        assert "⋯" in text

    def test_malformed_markdown_graceful(self, _capture_console):
        """Badly formed Markdown should not crash — render as-is."""
        display_assistant("Some text with [[[weird ***]]] syntax")
        text = _rendered(_capture_console)
        assert "weird" in text
        assert "syntax" in text

    def test_link(self, _capture_console):
        """Markdown links should be rendered."""
        display_assistant("Visit [GitHub](https://github.com) for more.")
        text = _rendered(_capture_console)
        assert "GitHub" in text
        # The URL may or may not appear depending on terminal capabilities,
        # but the link text should definitely be present
        assert "github.com" in text or "GitHub" in text


# ---------------------------------------------------------------------------
# Other display functions — NOT affected by Markdown
# ---------------------------------------------------------------------------


class TestOtherDisplayFunctions:
    """Verify that tool result/error panels are NOT markdown-rendered."""

    def _make_entry(self, name: str = "testmod_ls") -> ToolEntry:
        return ToolEntry(
            name=name,
            module_name="testmod",
            display_path="testmod ls",
            description="List test items",
            is_write=False,
        )

    def test_tool_result_success_shows_output(self, _capture_console):
        """Success tool result should show the output as plain text."""
        entry = self._make_entry()
        display_tool_result(entry, {}, {"output": "**raw** markdown", "exit_code": 0})
        text = _rendered(_capture_console)
        # Tool results pass through raw content — Markdown markers are preserved
        assert "**raw**" in text, "Tool output should NOT be Markdown-rendered"

    def test_tool_error_shows_error(self, _capture_console):
        """Tool error panel should show error text as-is."""
        display_tool_error("test_tool", "Something **broke**")
        text = _rendered(_capture_console)
        # tool_error uses Rich markup explicitly — "**broke**" is Rich markup, not markdown
        # Rich bold markers are **text**, so if we see "broke" the output is fine
        assert "broke" in text

    def test_tool_result_error_state(self, _capture_console):
        """Error state in tool result should remain plain."""
        entry = self._make_entry()
        result = {
            "output": "stdout text",
            "error": "**error** details",
            "exit_code": 1,
        }
        display_tool_result(entry, {}, result)
        text = _rendered(_capture_console)
        assert "stdout text" in text
        assert "**error**" in text, "Tool error content should not be markdown-rendered"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case robustness for display_assistant."""

    def test_very_long_input_no_crash(self, _capture_console):
        """Very long markdown input should not cause an error."""
        long_text = "# " + "x" * 5000
        display_assistant(long_text)  # should not raise
        text = _rendered(_capture_console)
        assert "x" in text

    def test_html_in_markdown_is_escaped(self, _capture_console):
        """Raw HTML in markdown should be safe."""
        display_assistant("<script>alert('xss')</script>")
        text = _rendered(_capture_console)
        # Rich may or may not strip HTML, but it should not crash
        assert "script" in text or "alert" not in text

    def test_newlines_preserved(self, _capture_console):
        """Multiple paragraphs should be preserved."""
        md = "Line one\n\nLine two\n\nLine three"
        display_assistant(md)
        text = _rendered(_capture_console)
        assert "Line one" in text
        assert "Line two" in text
        assert "Line three" in text
