"""Base types and helper functions for A-kunpiloto tool schemas.

Shared between registry (schema generation) and executor (execution).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, get_type_hints

from typer.models import (
    ArgumentInfo,
    DefaultPlaceholder,
    OptionInfo,
)


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


def is_write_command(command_name: str) -> bool:
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


def get_param_help(param_default: Any) -> str:
    """Extract help text from a parameter's default value.

    Args:
        param_default: The default value (may be ArgumentInfo, OptionInfo, etc.).

    Returns:
        Help text string.
    """
    if hasattr(param_default, "help") and param_default.help:
        return param_default.help
    return ""


def param_to_json_schema(
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
    if param_annotation is inspect.Parameter.empty:
        json_type = "string"
    else:
        json_type = CLICK_TYPE_MAP.get(param_annotation, "string")

    schema: dict[str, Any] = {"type": json_type}

    help_text = get_param_help(param_default)
    if help_text:
        schema["description"] = help_text

    if isinstance(param_default, DefaultPlaceholder):
        pass  # No meaningful default
    elif isinstance(param_default, (ArgumentInfo, OptionInfo)):
        pass  # Descriptors, not actual defaults
    elif param_default is not inspect.Parameter.empty:
        if json_type == "integer":
            schema["default"] = int(param_default)
        elif json_type == "number":
            schema["default"] = float(param_default)
        elif json_type == "boolean":
            schema["default"] = bool(param_default)
        else:
            schema["default"] = str(param_default)

    return schema


def is_positional_param(param: inspect.Parameter) -> bool:
    """Check if a parameter is a positional (Argument) parameter.

    Args:
        param: The inspect.Parameter object.

    Returns:
        True if the param is a positional argument.
    """
    if isinstance(param.default, ArgumentInfo):
        return True
    if isinstance(param.default, OptionInfo):
        return False
    if param.default is inspect.Parameter.empty:
        return True
    return False


def build_tool_schema(
    callback: Any,
    tool_name: str,
    module_name: str,
) -> tuple[dict[str, Any], list[str]]:
    """Build an OpenAI-compatible tool schema from a Typer callback function.

    Args:
        callback: The command callback function.
        tool_name: The canonical tool name.
        module_name: The A-module name (for context).

    Returns:
        A tuple of (schema_dict, positional_param_names).
    """
    properties: dict[str, Any] = {}
    required: list[str] = []
    positional: list[str] = []

    sig = inspect.signature(callback)
    type_hints = get_type_hints(callback)

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls", "ctx", "context"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        annotation = type_hints.get(param_name, str)
        prop_schema = param_to_json_schema(param_name, param.default, annotation)
        if prop_schema is None:
            continue

        properties[param_name] = prop_schema

        if is_positional_param(param):
            positional.append(param_name)
            required.append(param_name)

    description = ""
    if callback.__doc__:
        description = callback.__doc__.strip()

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
