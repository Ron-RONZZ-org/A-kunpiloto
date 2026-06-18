"""Tool registry — auto-discovers A-modules and builds LLM tool schemas.

Walks installed A-modules via ``A.commands`` entry points, inspects each
Typer app's registered commands and groups, and extracts tool definitions
in OpenAI's function-calling format using Typer 0.26+ native API.
"""

from __future__ import annotations

import importlib.metadata
import inspect
from dataclasses import dataclass, field
from typing import Any, get_type_hints

import typer
from typer.models import ArgumentInfo, CommandInfo, OptionInfo, TyperInfo


@dataclass
class ToolEntry:
    """A single tool wrapping a CLI command.

    Attributes:
        name: Canonical tool name (e.g. "vorto_aldoni").
        module_name: Module name (e.g. "vorto").
        display_path: Human-readable path (e.g. "vorto aldoni").
        description: Short description for the LLM.
        schema: OpenAI-compatible tool schema dict (the full function object).
        is_write: Whether this command modifies data.
        args_prefix: CLI arg prefix (e.g. ["retposto", "sendi"]).
        positional_params: Names of positional (Argument) params, in order.
    """
    name: str
    module_name: str
    display_path: str
    description: str
    schema: dict[str, Any] = field(default_factory=dict)
    is_write: bool = False
    args_prefix: list[str] = field(default_factory=list)
    positional_params: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Known write command name suffixes
# ---------------------------------------------------------------------------

WRITE_COMMAND_SUFFIXES: frozenset[str] = frozenset({
    "aldoni", "modifi", "forigi",
    "sendi", "respondi", "plusendi",
    "forgesi", "restauxrigi", "restaŭrigi", "malplenigi",
    "importi", "movi", "kreu", "krei",
})


def _is_write_command(command_name: str) -> bool:
    """Classify a command as write (destructive) or read-only.

    Args:
        command_name: The leaf command name (e.g. "aldoni").

    Returns:
        True if the command is a write operation.
    """
    return command_name in WRITE_COMMAND_SUFFIXES


# ---------------------------------------------------------------------------
# Schema generation helpers
# ---------------------------------------------------------------------------

CLICK_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _resolve_command_name(cmd_info: CommandInfo) -> str:
    """Get the effective command name from a CommandInfo.

    Args:
        cmd_info: The CommandInfo object.

    Returns:
        The command name string.
    """
    if cmd_info.name:
        return cmd_info.name
    if cmd_info.callback:
        return cmd_info.callback.__name__.replace("_", "-")
    return "unnamed"


def _get_param_help(param_default: Any) -> str:
    """Extract help text from a parameter's default value.

    Args:
        param_default: The default value (may be ArgumentInfo, OptionInfo, etc.).

    Returns:
        Help text string.
    """
    if hasattr(param_default, "help") and param_default.help:
        return param_default.help
    return ""


def _param_to_json_schema(
    param_name: str,
    param_default: Any,
    param_annotation: Any,
) -> dict[str, Any] | None:
    """Convert a function parameter to a JSON Schema property dict.

    Args:
        param_name: The parameter name.
        param_default: The parameter's default value.
        param_annotation: The type annotation.

    Returns:
        A JSON Schema property dict, or None if the param should be skipped.
    """
    # Determine JSON Schema type from annotation
    if param_annotation is inspect.Parameter.empty:
        json_type = "string"
    else:
        json_type = CLICK_TYPE_MAP.get(param_annotation, "string")

    schema: dict[str, Any] = {"type": json_type}

    # Help text from ArgumentInfo/OptionInfo
    help_text = _get_param_help(param_default)
    if help_text:
        schema["description"] = help_text

    # Default value (from OptionInfo or plain default)
    # Skip typer's internal DefaultPlaceholder
    from typer.models import DefaultPlaceholder
    if isinstance(param_default, DefaultPlaceholder):
        pass  # No meaningful default
    elif isinstance(param_default, (ArgumentInfo, OptionInfo)):
        pass  # These are descriptors, not actual defaults
    elif param_default is not inspect.Parameter.empty:
        # Plain default value (e.g. `count: int = 5`)
        if json_type == "integer":
            schema["default"] = int(param_default)
        elif json_type == "number":
            schema["default"] = float(param_default)
        elif json_type == "boolean":
            schema["default"] = bool(param_default)
        else:
            schema["default"] = str(param_default)

    return schema


def _is_positional_param(param: inspect.Parameter) -> bool:
    """Check if a parameter is a positional (Argument) parameter.

    In Typer 0.26+, parameters with ``typer.Argument(...)`` as default
    are positional. Parameters without defaults are also positional.

    Args:
        param: The inspect.Parameter object.

    Returns:
        True if the param is a positional argument.
    """
    # typer.Argument(...) → positional
    if isinstance(param.default, ArgumentInfo):
        return True
    # typer.Option(...) → not positional
    if isinstance(param.default, OptionInfo):
        return False
    # No default (and not *args, **kwargs) → positional
    if param.default is inspect.Parameter.empty:
        return True
    # Has a plain default value → option-style
    return False


def _build_tool_schema(
    callback: Any,
    tool_name: str,
    module_name: str,
) -> dict[str, Any]:
    """Build an OpenAI-compatible tool schema from a Typer callback function.

    Args:
        callback: The command callback function.
        tool_name: The canonical tool name.
        module_name: The A-module name (for context).

    Returns:
        A tool schema dict with parameters extracted from the function signature.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []
    positional: list[str] = []

    sig = inspect.signature(callback)
    type_hints = get_type_hints(callback)

    for param_name, param in sig.parameters.items():
        # Skip self, cls, context, *args, **kwargs
        if param_name in ("self", "cls", "ctx", "context"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        annotation = type_hints.get(param_name, str)
        prop_schema = _param_to_json_schema(param_name, param.default, annotation)
        if prop_schema is None:
            continue

        properties[param_name] = prop_schema

        is_positional = _is_positional_param(param)
        if is_positional:
            positional.append(param_name)
            required.append(param_name)

    # Description: docstring > help
    description = ""
    if callback.__doc__:
        description = callback.__doc__.strip()
    # Find help from CommandInfo (we don't have it here, but we'll use docstring)

    if not description:
        description = f"{module_name} {tool_name}"

    schema: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
            },
        },
    }

    if required:
        schema["function"]["parameters"]["required"] = required

    return schema, positional


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

    def get_schemas(self) -> list[dict[str, Any]]:
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

        Args:
            app: The Typer app to walk.
            module_name: The A-module name.
            prefix: Prefix for tool naming.
            args_prefix: CLI arg prefix for path-like subcommands.
        """
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
            self._walk_typer_app(
                app=sub_app,
                module_name=module_name,
                prefix=f"{prefix}_{name}",
                args_prefix=[*args_prefix, name],
            )

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
        full_name = f"{prefix}_{command_name}"

        if cmd_info.callback is None:
            return

        # Build schema and extract positional params
        schema, positional_params = _build_tool_schema(
            callback=cmd_info.callback,
            tool_name=full_name,
            module_name=module_name,
        )

        display_path = " ".join([module_name, *args_prefix, command_name])

        # Use CommandInfo help as description fallback
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
            is_write=_is_write_command(command_name),
            args_prefix=[*args_prefix, command_name],
            positional_params=positional_params,
        )

        self._entries[full_name] = entry
