"""Tests for tool name sanitization (diacritic stripping, ASCII safety)."""

from __future__ import annotations

import re

from A_kunpiloto.tools._base import normalize_tool_name

# OpenAI/DeepSeek pattern for valid function names
VALID_FN_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")


def _assert_valid(name: str) -> None:
    """Assert that a name matches the LLM function naming pattern."""
    assert VALID_FN_NAME.match(name), f"Invalid function name: {name!r}"


class TestNormalizeToolName:
    def test_ascii_passthrough(self):
        assert normalize_tool_name("vorto_aldoni") == "vorto_aldoni"

    def test_ascii_with_hyphens(self):
        assert normalize_tool_name("kunpiloto_repl") == "kunpiloto_repl"

    def test_esperanto_s_circumflex(self):
        """ŝ → s"""
        assert normalize_tool_name("sistemo_particio_ŝrumpi") == "sistemo_particio_srumpi"
        _assert_valid("sistemo_particio_srumpi")

    def test_esperanto_c_circumflex(self):
        """ĉ → c"""
        assert normalize_tool_name("vorto_serĉi") == "vorto_serci"
        _assert_valid("vorto_serci")

    def test_esperanto_g_circumflex(self):
        """ĝ → g"""
        assert normalize_tool_name("ebla_ĝisdatigi") == "ebla_gisdatigi"
        _assert_valid("ebla_gisdatigi")

    def test_esperanto_h_circumflex(self):
        """ĥ → h"""
        assert normalize_tool_name("iri_ĥaose") == "iri_haose"
        _assert_valid("iri_haose")

    def test_esperanto_j_circumflex(self):
        """ĵ → j"""
        assert normalize_tool_name("nova_ĵurnalo") == "nova_jurnalo"
        _assert_valid("nova_jurnalo")

    def test_esperanto_u_breve(self):
        """ŭ → u"""
        assert normalize_tool_name("restaŭri") == "restauri"
        _assert_valid("restauri")

    def test_mixed_diacritics(self):
        name = "sistemo_disko__ŝrumpi"
        result = normalize_tool_name(name)
        assert VALID_FN_NAME.match(result)
        assert "ŝ" not in result
        assert "__" not in result  # collapsed

    def test_punctuation_replaced(self):
        """Punctuation characters should be replaced with underscores."""
        result = normalize_tool_name("a.b.c")
        _assert_valid(result)
        assert "." not in result

    def test_regression_no_invalid_names(self):
        """All tool names from the full registry must be valid."""
        from A_kunpiloto.tools.registry import ToolRegistry
        reg = ToolRegistry()
        reg.build()
        for name in reg.tool_names:
            _assert_valid(name)
