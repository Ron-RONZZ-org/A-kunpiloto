"""Tool registry — auto-discovers A-modules and builds LLM tool schemas.

Walks installed A-modules via ``A.commands`` entry points, inspects each
Typer app's registered commands and groups, and extracts tool definitions
in OpenAI's function-calling format using Typer 0.26+ native API.
"""

from __future__ import annotations

import difflib
import importlib.metadata
from collections.abc import Callable
from typing import Any

import typer
from typer.models import CommandInfo, DefaultPlaceholder, TyperInfo

from A_kunpiloto.tools._base import (
    ToolEntry,
    build_tool_schema,
    is_write_command,
    normalize_tool_name,
)


def _get_app_help_text(app: typer.Typer) -> str:
    """Extract the module-level help text from a Typer app.

    Tries ``app.info.help`` first (the ``help=`` argument to ``Typer()``),
    then falls back to a default module description.

    Args:
        app: The Typer app instance.

    Returns:
        Help text string (may be empty).
    """
    info = getattr(app, "info", None)
    if info:
        raw = getattr(info, "help", None)
        if raw and not isinstance(raw, DefaultPlaceholder):
            return str(raw)
    return ""


def _make_module_tag(module_name: str) -> str:
    """Build a short module tag for tool descriptions.

    Args:
        module_name: The module name (e.g. "semantika").

    Returns:
        Tag string like "[semantika] ".
    """
    return f"[{module_name}] "


def _resolve_command_name(cmd_info: CommandInfo) -> str:
    """Get the effective command name from a CommandInfo.

    The returned name is normalized to ASCII to ensure it matches
    the OpenAI/DeepSeek function name pattern ``^[a-zA-Z0-9_-]+$``.

    Args:
        cmd_info: The CommandInfo object.

    Returns:
        The command name string (ASCII-safe).
    """
    raw = "unnamed"
    if cmd_info.name:
        raw = cmd_info.name
    elif cmd_info.callback:
        raw = cmd_info.callback.__name__
    return normalize_tool_name(raw)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Discovers installed A-modules and builds tool schemas.

    Usage::

        registry = ToolRegistry()
        registry.build()
        schemas = registry.get_schemas()
        entry = registry.get_entry("vorto_aldoni")
    """

    def __init__(self) -> None:
        self._entries: dict[str, ToolEntry] = {}
        self._module_names: list[str] = []
        # Module_name -> description (extracted from Typer app help text)
        self._module_descriptions: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Discover all A-modules and build tool entries.

        Scans ``A.commands`` entry points, inspects each Typer app's
        registered commands and sub-typer groups.
        """
        self._entries.clear()
        self._module_names.clear()
        self._module_descriptions.clear()

        for ep in importlib.metadata.entry_points(group="A.commands"):
            module_name = ep.name
            try:
                app = ep.load()
            except Exception:
                continue
            if not isinstance(app, typer.Typer):
                continue

            self._module_names.append(module_name)

            # Store module-level description from app help text
            app_help = _get_app_help_text(app)
            if app_help:
                self._module_descriptions[module_name] = app_help

            self._walk_typer_app(
                app=app,
                module_name=module_name,
                prefix=module_name,
                args_prefix=[],
            )

    def get_schemas(self) -> list[dict]:
        """Return all tool schemas for LLM consumption.

        Returns:
            List of OpenAI-compatible tool definition dicts.
        """
        return [entry.schema for entry in self._entries.values()]

    def get_entry(self, tool_name: str) -> ToolEntry | None:
        """Look up a tool entry by its canonical name.

        Args:
            tool_name: The tool name (e.g. "vorto_aldoni").

        Returns:
            The ToolEntry, or None if not found.
        """
        return self._entries.get(tool_name)

    def register_builtin(
        self,
        name: str,
        description: str,
        schema: dict[str, Any],
        handler: Callable[..., dict[str, Any]],
    ) -> None:
        """Register a built-in tool (not wrapping a CLI command).

        Args:
            name: Tool name (e.g. "read_result").
            description: Short description for the LLM.
            schema: Full OpenAI-compatible tool schema.
            handler: Callable[[dict], dict] that receives args and returns result.
        """
        entry = ToolEntry(
            name=name,
            module_name="_builtin",
            display_path=name,
            description=description,
            schema=schema,
            is_write=False,
            args_prefix=[],
            positional_params=[],
            handler=handler,
        )
        self._entries[name] = entry

    def find_similar_tools(self, name: str, cutoff: float = 0.5) -> list[str]:
        """Find registered tool names similar to *name*.

        Uses difflib fuzzy matching against all registered tool names.

        Args:
            name: The (possibly misspelled) tool name.
            cutoff: Similarity threshold (0.0–1.0).

        Returns:
            Up to 3 similar tool names.
        """
        matches = difflib.get_close_matches(
            name, self._entries.keys(), n=3, cutoff=cutoff,
        )
        return matches

    @property
    def module_names(self) -> list[str]:
        """List of discovered module names."""
        return list(self._module_names)

    @property
    def tool_names(self) -> list[str]:
        """List of all registered tool names."""
        return sorted(self._entries.keys())

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Internal: walk Typer app
    # ------------------------------------------------------------------

    def _walk_typer_app(
        self,
        app: typer.Typer,
        module_name: str,
        prefix: str,
        args_prefix: list[str],
    ) -> None:
        """Walk a Typer app's commands and sub-typers.

        Flat apps (no subcommands, root callback only like A-tempo) are
        detected and registered as a single tool.

        Args:
            app: The Typer app to walk.
            module_name: The A-module name.
            prefix: Prefix for tool naming.
            args_prefix: CLI arg prefix for path-like subcommands.
        """
        has_commands = bool(app.registered_commands)

        # Register leaf commands
        for cmd_info in app.registered_commands:
            if cmd_info.hidden:
                continue
            self._register_command(
                cmd_info=cmd_info,
                module_name=module_name,
                prefix=prefix,
                args_prefix=args_prefix,
            )

        # Recurse into sub-typer groups
        for group_info in app.registered_groups:
            if group_info.hidden:
                continue
            name = group_info.name or ""
            sub_app = group_info.typer_instance
            if sub_app is None:
                continue
            has_commands = True
            clean_name = normalize_tool_name(name)
            self._walk_typer_app(
                app=sub_app,
                module_name=module_name,
                prefix=f"{prefix}_{clean_name}",
                args_prefix=[*args_prefix, name],
            )

        # Flat app: no subcommands but has root callback with params
        if not has_commands and self._is_flat_app(app):
            self._register_flat_app_callback(
                app=app,
                module_name=module_name,
                prefix=prefix,
            )

    # ------------------------------------------------------------------
    # Command registration
    # ------------------------------------------------------------------

    def _register_command(
        self,
        cmd_info: CommandInfo,
        module_name: str,
        prefix: str,
        args_prefix: list[str],
    ) -> None:
        """Build and register a single tool entry from a CommandInfo.

        Args:
            cmd_info: The CommandInfo object.
            module_name: The A-module name.
            prefix: The full tool name prefix.
            args_prefix: CLI arg prefix.
        """
        command_name = _resolve_command_name(cmd_info)
        full_name = normalize_tool_name(f"{prefix}_{command_name}")

        if cmd_info.callback is None:
            return

        schema, positional_params = build_tool_schema(
            callback=cmd_info.callback,
            tool_name=full_name,
            module_name=module_name,
        )

        display_path = " ".join([module_name, *args_prefix, command_name])

        # Build enriched description with module context
        module_tag = _make_module_tag(module_name)
        module_desc = self._module_descriptions.get(module_name, "")
        # Priority: cmd_info.help > short_help > docstring from build_tool_schema
        base_desc = schema["function"]["description"]
        if cmd_info.help:
            base_desc = cmd_info.help
        elif cmd_info.short_help:
            base_desc = cmd_info.short_help

        if module_desc and base_desc:
            schema["function"]["description"] = (
                f"{module_tag}{module_desc} {base_desc}"
            )
        elif base_desc:
            schema["function"]["description"] = f"{module_tag}{base_desc}"
        elif module_desc:
            schema["function"]["description"] = module_tag
        # else: keep schema's existing description (from build_tool_schema)

        entry = ToolEntry(
            name=full_name,
            module_name=module_name,
            display_path=display_path,
            description=schema["function"]["description"],
            schema=schema,
            is_write=is_write_command(command_name),
            args_prefix=[*args_prefix, command_name],
            positional_params=positional_params,
        )

        self._entries[full_name] = entry

    # ------------------------------------------------------------------
    # Flat app support (no subcommands, e.g. A-tempo)
    # ------------------------------------------------------------------

    @staticmethod
    def _has_real_callback(app: typer.Typer) -> bool:
        """Check if the app has a real (non-default) root callback.

        Args:
            app: The Typer app.

        Returns:
            True if the app has a real callback function.
        """
        rc = getattr(app, "registered_callback", None)
        if rc is None:
            return False
        cb = getattr(rc, "callback", None)
        return cb is not None and not isinstance(cb, DefaultPlaceholder)

    def _is_flat_app(self, app: typer.Typer) -> bool:
        """Detect flat apps: no registered subcommands, but has root callback.

        Args:
            app: The Typer app.

        Returns:
            True if this is a flat app.
        """
        return (
            not app.registered_commands
            and not app.registered_groups
            and self._has_real_callback(app)
        )

    def _register_flat_app_callback(
        self,
        app: typer.Typer,
        module_name: str,
        prefix: str,
    ) -> None:
        """Register the root callback of a flat app as a tool.

        Args:
            app: The Typer app.
            module_name: The A-module name.
            prefix: The tool name prefix.
        """
        rc = getattr(app, "registered_callback", None)
        if rc is None:
            return
        callback = getattr(rc, "callback", None)
        if callback is None:
            return

        tool_name = normalize_tool_name(prefix)
        schema, positional_params = build_tool_schema(
            callback=callback,
            tool_name=tool_name,
            module_name=module_name,
        )

        # Build enriched description with module context
        module_tag = _make_module_tag(module_name)
        module_desc = self._module_descriptions.get(module_name, "")
        # Priority: rc.help > callback docstring > schema default from build_tool_schema
        base_desc = schema["function"]["description"]
        if rc.help and not isinstance(rc.help, DefaultPlaceholder):
            base_desc = rc.help
        elif callback.__doc__:
            base_desc = callback.__doc__.strip()

        if module_desc and base_desc:
            schema["function"]["description"] = (
                f"{module_tag}{module_desc} {base_desc}"
            )
        elif base_desc:
            schema["function"]["description"] = f"{module_tag}{base_desc}"
        elif module_desc:
            schema["function"]["description"] = module_tag
        # else: keep schema's existing description

        entry = ToolEntry(
            name=tool_name,
            module_name=module_name,
            display_path=module_name,
            description=schema["function"]["description"],
            schema=schema,
            is_write=is_write_command(tool_name),
            args_prefix=[],
            positional_params=positional_params,
        )
        self._entries[tool_name] = entry
