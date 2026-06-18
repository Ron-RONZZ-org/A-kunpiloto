"""Tests for A-kunpiloto safety gate."""

from __future__ import annotations

from unittest.mock import patch

from A_kunpiloto.tools.safety import classify_command


class TestClassifyCommand:
    def test_aldoni_is_write(self):
        assert classify_command("testmod_aldoni") == "write"

    def test_modifi_is_write(self):
        assert classify_command("testmod_modifi") == "write"

    def test_forigi_is_write(self):
        assert classify_command("testmod_forigi") == "write"

    def test_sendi_is_write(self):
        assert classify_command("lien_retposto_sendi") == "write"

    def test_ls_is_read(self):
        assert classify_command("testmod_ls") == "read"

    def test_vidi_is_read(self):
        assert classify_command("testmod_vidi") == "read"

    def test_serci_is_read(self):
        assert classify_command("testmod_serci") == "read"

    def test_unknown_is_unknown(self):
        assert classify_command("testmod_nekonata") == "unknown"
        assert classify_command("foo_bar_baz") == "unknown"

    def test_empty_string(self):
        assert classify_command("") == "unknown"

    def test_write_preview_shows_correctly(self, registry):
        """Preview table should render without errors."""
        from A_kunpiloto.tools.safety import _build_preview_table

        entry = registry.get_entry("testmod_aldoni")
        assert entry is not None
        panel = _build_preview_table(entry, {"nomo": "test", "kategoria": "gen"})
        assert panel is not None
        # Panel title should contain the module name and command
        title_html = str(panel.title)
        assert "testmod" in title_html
        assert "aldoni" in title_html

    def test_confirm_write_returns_true_on_yes(self, registry):
        """User typing 'y' should return True."""
        from A_kunpiloto.tools.safety import confirm_write_operation

        entry = registry.get_entry("testmod_aldoni")
        assert entry is not None

        with patch("builtins.input", return_value="y"):
            assert confirm_write_operation(entry, {"nomo": "test"}) is True

    def test_confirm_write_returns_false_on_no(self, registry):
        """User typing 'n' should return False."""
        from A_kunpiloto.tools.safety import confirm_write_operation

        entry = registry.get_entry("testmod_aldoni")
        assert entry is not None

        with patch("builtins.input", return_value="n"):
            assert confirm_write_operation(entry, {"nomo": "test"}) is False

    def test_confirm_write_returns_false_on_empty(self, registry):
        """Empty input (default N) should return False."""
        from A_kunpiloto.tools.safety import confirm_write_operation

        entry = registry.get_entry("testmod_aldoni")
        assert entry is not None

        with patch("builtins.input", return_value=""):
            assert confirm_write_operation(entry, {"nomo": "test"}) is False
