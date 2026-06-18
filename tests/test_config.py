"""Tests for A-kunpiloto configuration (system prompt file loading)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from A_kunpiloto.config import (
    DEFAULT_SYSTEM_PROMPT,
    commands_dir,
    config_path,
    ensure_config,
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


# ══════════════════════════════════════════════════════════════════════════
# Config file auto-seeding tests
# ══════════════════════════════════════════════════════════════════════════


class TestConfigPath:
    def test_config_path_under_kunpiloto(self, monkeypatch, tmp_path):
        monkeypatch.setenv("A_DIR", str(tmp_path))
        cp = config_path()
        assert "kunpiloto" in str(cp)
        assert cp.name == "config.toml"

    def test_config_path_ends_correctly(self, monkeypatch, tmp_path):
        monkeypatch.setenv("A_DIR", str(tmp_path))
        cp = config_path()
        assert str(cp).endswith("kunpiloto/config.toml")


class TestEnsureConfig:
    def test_seeds_config_when_nonexistent(self, monkeypatch, tmp_path):
        """Should create config.toml from shipped defaults when none exists."""
        monkeypatch.setenv("A_DIR", str(tmp_path))
        result = ensure_config()
        assert result is not None
        assert result.exists()
        assert result.name == "config.toml"
        content = result.read_text(encoding="utf-8")
        assert "write_always_allowed_directories" in content
        assert "/tmp/A/kunpiloto/**" in content

    def test_does_not_overwrite_existing_config(self, monkeypatch, tmp_path):
        """Should NOT overwrite a config the user has edited."""
        monkeypatch.setenv("A_DIR", str(tmp_path))
        cp = config_path()
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text('write_always_allowed_directories = ["/custom/path/**"]\n', encoding="utf-8")
        original_content = cp.read_text(encoding="utf-8")

        ensure_config()
        assert cp.read_text(encoding="utf-8") == original_content
        assert "/custom/path" in cp.read_text(encoding="utf-8")

    def test_seeded_config_is_valid_toml(self, monkeypatch, tmp_path):
        """The seeded config.toml must parse as valid TOML."""
        import tomllib
        monkeypatch.setenv("A_DIR", str(tmp_path))
        ensure_config()
        cp = config_path()
        with open(cp, "rb") as f:
            data = tomllib.load(f)
        assert "write_always_allowed_directories" in data
        assert isinstance(data["write_always_allowed_directories"], list)
        assert "read_always_allowed_directories" in data
        assert isinstance(data["read_always_allowed_directories"], list)

    def test_seeded_write_allowlist_has_temp_dir(self, monkeypatch, tmp_path):
        """Default write allowlist must include /tmp/A/kunpiloto/**."""
        monkeypatch.setenv("A_DIR", str(tmp_path))
        ensure_config()
        cp = config_path()
        with open(cp, "rb") as f:
            import tomllib
            data = tomllib.load(f)
        assert "/tmp/A/kunpiloto/**" in data["write_always_allowed_directories"]

    def test_seeded_read_allowlist_has_tmp(self, monkeypatch, tmp_path):
        """Default read allowlist must include /tmp/**."""
        monkeypatch.setenv("A_DIR", str(tmp_path))
        ensure_config()
        cp = config_path()
        with open(cp, "rb") as f:
            import tomllib
            data = tomllib.load(f)
        assert "/tmp/**" in data["read_always_allowed_directories"]

    def test_ensure_config_creates_parent_dir(self, monkeypatch, tmp_path):
        """Parent directory should be created automatically."""
        monkeypatch.setenv("A_DIR", str(tmp_path))
        ensure_config()
        assert config_path().parent.is_dir()

    def test_ensure_config_also_seeds_system_prompt(self, monkeypatch, tmp_path):
        """ensure_config should also seed the system prompt."""
        monkeypatch.setenv("A_DIR", str(tmp_path))
        ensure_config()
        assert system_prompt_path().exists()
