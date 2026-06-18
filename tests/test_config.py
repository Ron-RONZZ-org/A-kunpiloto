"""Tests for A-kunpiloto configuration (system prompt file loading)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from A_kunpiloto.config import (
    DEFAULT_SYSTEM_PROMPT,
    commands_dir,
    load_system_prompt,
    system_prompt_path,
)


def _create_system_prompt(tmp_path: Path, content: str) -> Path:
    """Create a system_prompt.md under tmp_path and return its path."""
    prompt_dir = tmp_path / "config" / "kunpiloto"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    path = prompt_dir / "system_prompt.md"
    path.write_text(content, encoding="utf-8")
    return path


class TestSystemPromptPath:
    def test_system_prompt_path_under_kunpiloto(self, monkeypatch, tmp_path):
        monkeypatch.setenv("A_DIR", str(tmp_path))
        sp_path = system_prompt_path()
        assert "kunpiloto" in str(sp_path)
        assert sp_path.name == "system_prompt.md"

    def test_system_prompt_path_trailing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("A_DIR", str(tmp_path))
        sp_path = system_prompt_path()
        assert str(sp_path).endswith("kunpiloto/system_prompt.md")

    def test_commands_dir_under_kunpiloto(self, monkeypatch, tmp_path):
        monkeypatch.setenv("A_DIR", str(tmp_path))
        cd = commands_dir()
        assert "kunpiloto" in str(cd)
        assert cd.name == "commands"


class TestLoadSystemPrompt:
    def test_returns_default_when_no_file(self, monkeypatch, tmp_path):
        """Without any file or shipped resource, must fall back to hardcoded default."""
        monkeypatch.setenv("A_DIR", str(tmp_path))
        with patch("A_kunpiloto.config._read_shipped", return_value=None):
            prompt = load_system_prompt()
        assert prompt == DEFAULT_SYSTEM_PROMPT

    def test_returns_file_content_when_exists(self, monkeypatch, tmp_path):
        monkeypatch.setenv("A_DIR", str(tmp_path))
        _create_system_prompt(tmp_path, "You are a test bot.")
        prompt = load_system_prompt()
        assert prompt == "You are a test bot."

    def test_strips_whitespace(self, monkeypatch, tmp_path):
        monkeypatch.setenv("A_DIR", str(tmp_path))
        _create_system_prompt(tmp_path, "  \n  Hello World  \n  ")
        prompt = load_system_prompt()
        assert prompt == "Hello World"

    def test_returns_default_when_file_is_empty(self, monkeypatch, tmp_path):
        monkeypatch.setenv("A_DIR", str(tmp_path))
        _create_system_prompt(tmp_path, "   \n  \n  ")
        with patch("A_kunpiloto.config._read_shipped", return_value=None):
            prompt = load_system_prompt()
        assert prompt == DEFAULT_SYSTEM_PROMPT

    def test_returns_default_when_only_whitespace(self, monkeypatch, tmp_path):
        monkeypatch.setenv("A_DIR", str(tmp_path))
        _create_system_prompt(tmp_path, "\t  \n  \n  ")
        with patch("A_kunpiloto.config._read_shipped", return_value=None):
            prompt = load_system_prompt()
        assert prompt == DEFAULT_SYSTEM_PROMPT

    def test_multiline_content(self, monkeypatch, tmp_path):
        monkeypatch.setenv("A_DIR", str(tmp_path))
        content = "Line 1\nLine 2\nLine 3"
        _create_system_prompt(tmp_path, content)
        prompt = load_system_prompt()
        assert prompt == content

    def test_returns_file_over_default(self, monkeypatch, tmp_path):
        """A real file must take priority over the hardcoded default."""
        monkeypatch.setenv("A_DIR", str(tmp_path))
        _create_system_prompt(tmp_path, "Custom prompt")
        prompt = load_system_prompt()
        assert prompt != DEFAULT_SYSTEM_PROMPT
        assert prompt == "Custom prompt"

    def test_auto_seeds_shipped_default_on_first_run(self, monkeypatch, tmp_path):
        """When no user file exists, the shipped default should be auto-seeded."""
        monkeypatch.setenv("A_DIR", str(tmp_path))
        prompt = load_system_prompt()
        # Should return the shipped content, not the hardcoded fallback
        assert prompt != DEFAULT_SYSTEM_PROMPT
        assert len(prompt) > 100
        # Should have created the file on disk
        assert system_prompt_path().exists()
        disk_content = system_prompt_path().read_text(encoding="utf-8").strip()
        assert disk_content == prompt

    def test_auto_seed_creates_config_dir(self, monkeypatch, tmp_path):
        """The config directory should be created when auto-seeding."""
        monkeypatch.setenv("A_DIR", str(tmp_path))
        load_system_prompt()
        assert system_prompt_path().parent.is_dir()
