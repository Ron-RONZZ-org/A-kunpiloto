"""Tests for A-kunpiloto built-in file tools (write_file, read_file)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from A_kunpiloto.tools.file_tools import (
    READ_FILE_SCHEMA,
    WRITE_FILE_SCHEMA,
    _content_preview,
    _has_glob,
    _is_text,
    handle_read_file,
    handle_write_file,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def config() -> dict:
    """Default config for file tools."""
    return {
        "write_always_allowed_directories": [],
        "write_max_file_size": 10 * 1024 * 1024,  # 10 MB
        "read_always_allowed_directories": [],
    }


@pytest.fixture
def permissive_config(tmp_path: Path) -> dict:
    """Config that allows writing to tmp_path."""
    return {
        "write_always_allowed_directories": [str(tmp_path / "**")],
        "write_max_file_size": 10 * 1024 * 1024,  # 10 MB
        "read_always_allowed_directories": [],
    }


# ── Schema tests ──────────────────────────────────────────────────────────


class TestWriteFileSchema:
    def test_schema_has_required_fields(self):
        props = WRITE_FILE_SCHEMA["function"]["parameters"]["properties"]
        assert "file_path" in props
        assert "content" in props
        assert "mode" in props
        assert WRITE_FILE_SCHEMA["function"]["parameters"]["required"] == [
            "file_path",
            "content",
        ]

    def test_schema_mode_enum(self):
        """Mode should be an enum with overwrite and append."""
        mode_prop = WRITE_FILE_SCHEMA["function"]["parameters"]["properties"]["mode"]
        assert mode_prop.get("enum") == ["overwrite", "append"]


class TestReadFileSchema:
    def test_schema_has_required_fields(self):
        props = READ_FILE_SCHEMA["function"]["parameters"]["properties"]
        assert "file_path" in props
        assert "start_line" in props
        assert "end_line" in props
        assert READ_FILE_SCHEMA["function"]["parameters"]["required"] == ["file_path"]


# ── Handler tests: write_file ─────────────────────────────────────────────


class TestHandleWriteFile:
    def test_write_file_requires_file_path(self, config):
        result = handle_write_file({"file_path": "", "content": "hello"}, config)
        assert result["exit_code"] == 1
        assert "file_path is required" in result["error"]

    def test_write_file_invalid_mode(self, config):
        result = handle_write_file(
            {"file_path": "/tmp/test.txt", "content": "hello", "mode": "invalid"},
            config,
        )
        assert result["exit_code"] == 1
        assert "Invalid mode" in result["error"]

    def test_write_file_rejects_binary(self, config):
        result = handle_write_file(
            {"file_path": "/tmp/test.txt", "content": "hello\x00world"},
            config,
        )
        assert result["exit_code"] == 1
        assert "text content only" in result["error"].lower()

    def test_write_file_rejects_too_large(self, config):
        config["write_max_file_size"] = 5
        result = handle_write_file(
            {"file_path": "/tmp/test.txt", "content": "x" * 10},
            config,
        )
        assert result["exit_code"] == 1
        assert "too large" in result["error"].lower()

    def test_write_file_requires_confirmation_if_not_allowed(self, config):
        """Without allowlist and user declines, write is cancelled."""
        with patch(
            "A_kunpiloto.tools.file_tools.confirm_file_access",
            return_value=False,
        ):
            result = handle_write_file(
                {"file_path": "/tmp/test_write.txt", "content": "hello"},
                config,
            )
        assert result["output"] == "Write cancelled by user."
        assert result["exit_code"] == 0

    def test_write_file_to_allowed_path_auto_allow(self, tmp_path, permissive_config):
        """When path is in allowlist, no confirmation needed."""
        target = tmp_path / "sub" / "test.txt"
        result = handle_write_file(
            {"file_path": str(target), "content": "hello world"},
            permissive_config,
        )
        assert result["exit_code"] == 0
        assert target.read_text() == "hello world"

    def test_write_file_append_mode(self, tmp_path, permissive_config):
        """Append mode adds content without overwriting."""
        target = tmp_path / "append_test.txt"
        target.write_text("line1\n")
        result = handle_write_file(
            {
                "file_path": str(target),
                "content": "line2\n",
                "mode": "append",
            },
            permissive_config,
        )
        assert result["exit_code"] == 0
        assert target.read_text() == "line1\nline2\n"

    def test_write_file_overwrite_mode(self, tmp_path, permissive_config):
        """Overwrite mode replaces existing content."""
        target = tmp_path / "overwrite_test.txt"
        target.write_text("old content")
        result = handle_write_file(
            {
                "file_path": str(target),
                "content": "new content",
                "mode": "overwrite",
            },
            permissive_config,
        )
        assert result["exit_code"] == 0
        assert target.read_text() == "new content"

    def test_write_file_creates_parent_dirs(self, tmp_path, permissive_config):
        """Parent directories are created automatically."""
        target = tmp_path / "a" / "b" / "deep.txt"
        result = handle_write_file(
            {"file_path": str(target), "content": "deep"},
            permissive_config,
        )
        assert result["exit_code"] == 0
        assert target.exists()
        assert target.read_text() == "deep"


# ── Handler tests: read_file ──────────────────────────────────────────────


class TestHandleReadFile:
    def test_read_file_requires_file_path(self, config):
        result = handle_read_file({"file_path": ""}, config)
        assert result["exit_code"] == 1
        assert "file_path is required" in result["error"]

    def test_read_file_not_found(self, config):
        with patch(
            "A_kunpiloto.tools.file_tools.confirm_file_access",
            return_value=True,
        ):
            result = handle_read_file({"file_path": "/tmp/nonexistent-file-12345.txt"}, config)
        assert result["exit_code"] == 1
        assert "not found" in result["error"].lower() or "File not found" in result["error"]

    def test_read_file_requires_confirmation_if_not_allowed(self, config):
        """Without allowlist and user declines, read is cancelled."""
        with patch(
            "A_kunpiloto.tools.file_tools.confirm_file_access",
            return_value=False,
        ):
            result = handle_read_file(
                {"file_path": "/tmp/test_read_nope.txt"},
                config,
            )
        assert result["output"] == "Read cancelled by user."
        assert result["exit_code"] == 0

    def test_read_file_auto_allowed_through_write_allowlist(self, tmp_path, permissive_config):
        """Write allowlist directories are auto-added to read permissions."""
        target = tmp_path / "auto_read.txt"
        target.write_text("auto read test")
        result = handle_read_file(
            {"file_path": str(target)},
            permissive_config,
        )
        assert result["exit_code"] == 0
        assert "auto read test" in result["output"]

    def test_read_file_full_content(self, tmp_path):
        """Reading the full file returns all lines."""
        target = tmp_path / "full.txt"
        content = "line1\nline2\nline3\n"
        target.write_text(content)

        # Use empty config but mock confirmation to bypass safety
        with patch(
            "A_kunpiloto.tools.file_tools.confirm_file_access",
            return_value=True,
        ):
            result = handle_read_file(
                {"file_path": str(target)},
                {},
            )
        assert result["exit_code"] == 0
        assert "line1" in result["output"]
        assert "line2" in result["output"]
        assert "line3" in result["output"]
        assert "Total lines: 3" in result["output"]

    def test_read_file_line_range(self, tmp_path):
        """Reading a specific line range works."""
        target = tmp_path / "range.txt"
        target.write_text("\n".join(f"line{i}" for i in range(1, 101)))

        with patch(
            "A_kunpiloto.tools.file_tools.confirm_file_access",
            return_value=True,
        ):
            result = handle_read_file(
                {
                    "file_path": str(target),
                    "start_line": 5,
                    "end_line": 7,
                },
                {},
            )
        assert result["exit_code"] == 0
        assert "line5" in result["output"]
        assert "line6" in result["output"]
        assert "line7" in result["output"]
        assert "Showing lines 5" in result["output"]

    def test_read_file_caps_at_500_lines(self, tmp_path):
        """Reading more than 500 lines is capped."""
        target = tmp_path / "large.txt"
        target.write_text("\n".join(f"line{i}" for i in range(1, 1001)))

        with patch(
            "A_kunpiloto.tools.file_tools.confirm_file_access",
            return_value=True,
        ):
            result = handle_read_file(
                {
                    "file_path": str(target),
                    "start_line": 1,
                    "end_line": 1000,
                },
                {},
            )
        assert result["exit_code"] == 0
        # Should show at most 500 lines
        line_count = result["output"].count("\n")
        assert line_count <= 505  # header lines + content

    def test_read_file_binary_rejected(self, tmp_path):
        """Reading a binary file is rejected."""
        target = tmp_path / "binary.bin"
        target.write_bytes(b"\x00\x01\x02")

        with patch(
            "A_kunpiloto.tools.file_tools.confirm_file_access",
            return_value=True,
        ):
            result = handle_read_file(
                {"file_path": str(target)},
                {},
            )
        assert result["exit_code"] == 1
        assert "binary" in result["error"].lower()


# ── Helper tests ──────────────────────────────────────────────────────────


class TestIsText:
    def test_text_is_valid(self):
        assert _is_text("hello world") is True
        assert _is_text("") is True
        assert _is_text("héllo wörld 🎉") is True

    def test_null_byte_detected(self):
        assert _is_text("hello\x00world") is False


class TestHasGlob:
    def test_detects_glob_chars(self):
        assert _has_glob("/tmp/**") is True
        assert _has_glob("/tmp/*") is True
        assert _has_glob("/tmp/?") is True
        assert _has_glob("/tmp/[abc]") is True

    def test_rejects_plain_paths(self):
        assert _has_glob("/tmp/foo.txt") is False


class TestContentPreview:
    def test_short_content(self):
        preview = _content_preview("hello world")
        assert "1 lines" in preview
        assert "hello world" in preview

    def test_truncates_long_content(self):
        long = "x" * 500
        preview = _content_preview(long, max_chars=50)
        # The preview shows total content length (500 chars) and truncation
        assert "500 chars" in preview
        assert "..." in preview
        assert len(preview) < 100  # preview itself should be short

    def test_shows_line_count(self):
        preview = _content_preview("a\nb\nc\n")
        assert "4 lines" in preview
