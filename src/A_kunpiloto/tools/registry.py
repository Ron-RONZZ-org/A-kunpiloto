"""Tool registry — auto-discovers A-modules and builds LLM tool schemas.

Walks installed A-modules via ``A.commands`` entry points, inspects each
Typer app's registered commands and groups, and extracts tool definitions
in OpenAI's function-calling format using Typer 0.26+ native API.
"""

from __future__ import annotations

import importlib.metadata

import typer
from typer.models import CommandInfo, DefaultPlaceholder, TyperInfo

from A_kunpiloto.tools._base import (
    ToolEntry,
    build_tool_schema,
    is_write_command,
    normalize_tool_name,
)


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

        for ep in importlib.metadata.entry_points(group="A.commands"):
            module_name = ep.name
            try:
                app = ep.load()
            except Exception:
                continue
            if not isinstance(app, typer.Typer):
                continue

            self._module_names.append(module_name)
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

        if cmd_info.help:
            schema["function"]["description"] = cmd_info.help
        elif cmd_info.short_help:
            schema["function"]["description"] = cmd_info.short_help

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

        if callback.__doc__:
            schema["function"]["description"] = callback.__doc__.strip()
        elif rc.help and not isinstance(rc.help, DefaultPlaceholder):
            schema["function"]["description"] = rc.help

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
