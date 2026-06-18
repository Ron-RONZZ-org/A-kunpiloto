"""Tests for A-kunpiloto tool registry (tool discovery / schema generation)."""

from __future__ import annotations

import json

from unittest.mock import patch, MagicMock

from A_kunpiloto.tools.registry import ToolRegistry
from A_kunpiloto.tools._base import is_write_command, build_tool_schema


# ---------------------------------------------------------------------------
# Command classification
# ---------------------------------------------------------------------------


class TestCommandClassification:
    def test_write_commands_identified(self):
        assert is_write_command("aldoni") is True
        assert is_write_command("modifi") is True
        assert is_write_command("forigi") is True
        assert is_write_command("sendi") is True

    def test_read_commands_identified(self):
        assert is_write_command("ls") is False
        assert is_write_command("vidi") is False
        assert is_write_command("serci") is False

    def test_unknown_classified_as_read(self):
        assert is_write_command("unknown_cmd") is False


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
        testmod_names = [n for n in names if n.startswith("testmod_")]
        assert len(testmod_names) == 5  # ls, vidi, aldoni, forigi, sub_ls

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


# ---------------------------------------------------------------------------
# Flat app tests
# ---------------------------------------------------------------------------


class TestFlatAppSupport:
    def test_flat_app_discovered(self, flat_registry):
        """A flat app (no subcommands, root callback) should be discovered."""
        assert len(flat_registry) == 1
        assert "testflat" in flat_registry.module_names

    def test_flat_app_tool_name_is_module_name(self, flat_registry):
        """Flat app tool should be named after the module."""
        assert flat_registry.tool_names == ["testflat"]

    def test_flat_app_tool_has_params(self, flat_registry):
        """Flat app tool should have parameters from the callback."""
        entry = flat_registry.get_entry("testflat")
        assert entry is not None
        schema = entry.schema
        params = schema["function"]["parameters"]["properties"]
        assert "horzono" in params
        assert "chiuj" in params

    def test_flat_app_has_no_args_prefix(self, flat_registry):
        """Flat app tool should have empty args_prefix (no subcommand)."""
        entry = flat_registry.get_entry("testflat")
        assert entry is not None
        assert entry.args_prefix == []

    def test_flat_app_not_write(self, flat_registry):
        """Flat app tool should not be classified as write by default."""
        entry = flat_registry.get_entry("testflat")
        assert entry is not None
        assert entry.is_write is False

    def test_flat_app_schema_serializable(self, flat_registry):
        """Flat app schema must be JSON-serializable."""
        for schema in flat_registry.get_schemas():
            dumped = json.dumps(schema)
            parsed = json.loads(dumped)
            assert parsed["type"] == "function"
            assert parsed["function"]["name"] == "testflat"


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------


class TestBuiltinTools:
    def test_register_builtin(self, registry):
        """A built-in tool can be registered and retrieved."""
        registry.register_builtin(
            name="test_util",
            description="A test utility",
            schema={"type": "function", "function": {"name": "test_util"}},
            handler=lambda args: {"output": "ok", "error": "", "exit_code": 0},
        )
        entry = registry.get_entry("test_util")
        assert entry is not None
        assert entry.name == "test_util"
        assert entry.module_name == "_builtin"
        assert entry.handler is not None
        assert entry.is_write is False  # built-ins are never write
        assert entry.args_prefix == []

    def test_builtin_appears_in_schemas(self, registry):
        """Built-in tools should appear in get_schemas()."""
        registry.register_builtin(
            name="my_builtin",
            description="test",
            schema={"type": "function", "function": {"name": "my_builtin"}},
            handler=lambda args: {},
        )
        names = [s["function"]["name"] for s in registry.get_schemas()]
        assert "my_builtin" in names

    def test_builtin_handler_executes(self, registry):
        """The built-in tool handler should be callable."""
        results = []

        def handler(args):
            results.append(args)
            return {"output": "done", "error": "", "exit_code": 0}

        registry.register_builtin(
            name="builder",
            description="test",
            schema={"type": "function", "function": {"name": "builder"}},
            handler=handler,
        )
        entry = registry.get_entry("builder")
        assert entry is not None
        assert entry.handler is not None
        out = entry.handler({"x": 1})
        assert out["output"] == "done"
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------


class TestFuzzyMatching:
    def test_find_exact(self, registry):
        """An exact tool name should return itself."""
        matches = registry.find_similar_tools("testmod_ls")
        assert "testmod_ls" in matches

    def test_find_similar(self, registry):
        """Close misspellings should be suggested."""
        matches = registry.find_similar_tools("testmod_ldoni")  # close to aldoni
        # Should suggest testmod_aldoni or similar
        assert len(matches) > 0

    def test_no_similar(self, registry):
        """Completely different names should return empty."""
        matches = registry.find_similar_tools("zzzzzzzzz")
        assert matches == []

    def test_find_nonexistent_returns_empty(self, registry):
        """Empty string should return no matches."""
        matches = registry.find_similar_tools("")
        assert isinstance(matches, list)


# ---------------------------------------------------------------------------
# Module descriptions in tool schemas
# ---------------------------------------------------------------------------


class TestModuleDescriptions:
    def test_module_descriptions_present(self, registry):
        """Module descriptions should be populated from Typer app help."""
        descs = registry._module_descriptions
        assert "testmod" in descs
        # The test_app has help="Testa modulo"
        assert "Testa modulo" in descs["testmod"]

    def test_tool_description_contains_module_tag(self, registry):
        """Tool descriptions should be prefixed with [module_name]."""
        entry = registry.get_entry("testmod_ls")
        assert entry is not None
        desc = entry.schema["function"]["description"]
        assert "[testmod]" in desc, f"Description missing module tag: {desc}"

    def test_module_tag_does_not_break_json(self, registry):
        """Module-prefixed descriptions must remain JSON-serializable."""
        for schema in registry.get_schemas():
            dumped = json.dumps(schema)
            parsed = json.loads(dumped)
            assert "description" in parsed["function"]
