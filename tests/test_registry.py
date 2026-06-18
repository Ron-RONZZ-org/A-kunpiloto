"""Tests for A-kunpiloto tool registry (tool discovery / schema generation)."""

from __future__ import annotations

import json

from unittest.mock import patch, MagicMock

from A_kunpiloto.tools.registry import ToolRegistry, _is_write_command, _build_tool_schema


# ---------------------------------------------------------------------------
# Command classification
# ---------------------------------------------------------------------------


class TestCommandClassification:
    def test_write_commands_identified(self):
        assert _is_write_command("aldoni") is True
        assert _is_write_command("modifi") is True
        assert _is_write_command("forigi") is True
        assert _is_write_command("sendi") is True

    def test_read_commands_identified(self):
        assert _is_write_command("ls") is False
        assert _is_write_command("vidi") is False
        assert _is_write_command("serci") is False

    def test_unknown_classified_as_read(self):
        assert _is_write_command("unknown_cmd") is False


# ---------------------------------------------------------------------------
# Tool Discovery & Schema
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_empty_registry(self):
        reg = ToolRegistry()
        assert len(reg) == 0
        assert reg.tool_names == []

    def test_discovery_from_entry_points(self, mock_entry_points):
        with patch("importlib.metadata.entry_points", return_value=mock_entry_points):
            reg = ToolRegistry()
            reg.build()

        assert len(reg) > 0
        assert "testmod" in reg.module_names

    def test_tool_names_contain_module_prefix(self, registry):
        names = registry.tool_names
        assert all(n.startswith("testmod_") for n in names)

    def test_tool_entry_has_required_fields(self, registry):
        entry = registry.get_entry("testmod_ls")
        assert entry is not None
        assert entry.name == "testmod_ls"
        assert entry.module_name == "testmod"
        assert entry.is_write is False

        # aldoni should be write
        entry2 = registry.get_entry("testmod_aldoni")
        assert entry2 is not None
        assert entry2.is_write is True

    def test_schema_has_correct_structure(self, registry):
        entry = registry.get_entry("testmod_ls")
        schema = entry.schema
        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == "testmod_ls"
        assert "parameters" in schema["function"]

    def test_schema_has_required_params(self, registry):
        entry = registry.get_entry("testmod_aldoni")
        schema = entry.schema
        params = schema["function"]["parameters"]
        # "nomo" is a required Argument
        assert "nomo" in params["properties"]
        assert "nomo" in params.get("required", [])

    def test_schema_optional_params_not_required(self, registry):
        entry = registry.get_entry("testmod_ls")
        schema = entry.schema
        params = schema["function"]["parameters"]
        # "kategoria" is optional
        if "kategoria" in params.get("properties", {}):
            if "required" in params:
                assert "kategoria" not in params["required"]

    def test_schemas_serializable_to_json(self, registry):
        """All schemas must be JSON-serializable for LLM consumption."""
        for schema in registry.get_schemas():
            dumped = json.dumps(schema)
            parsed = json.loads(dumped)
            assert parsed["type"] == "function"

    def test_sub_typer_commands_discovered(self, registry):
        """Commands under a sub-typer should be discovered."""
        names = registry.tool_names
        assert any("sub" in n for n in names), f"No sub-typer tools found in {names}"

    def test_unknown_tool_returns_none(self, registry):
        assert registry.get_entry("nonexistent_tool") is None

    def test_entry_points_load_error_skipped(self):
        """Modules that fail to load should be skipped, not crash."""
        good_ep = MagicMock()
        good_ep.name = "good"
        good_ep.load.return_value = MagicMock(spec=["registered_commands"])

        bad_ep = MagicMock()
        bad_ep.name = "bad"
        bad_ep.load.side_effect = ImportError("broken")

        with patch("importlib.metadata.entry_points", return_value=[bad_ep, good_ep]):
            reg = ToolRegistry()
            reg.build()  # should not raise
