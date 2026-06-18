"""Base types and helper functions for A-kunpiloto tool schemas.

Shared between registry (schema generation) and executor (execution).
"""

from __future__ import annotations

import inspect
import re
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


# ---------------------------------------------------------------------------
# Tool name sanitization
# ---------------------------------------------------------------------------

# Mapping of non-ASCII characters to ASCII equivalents for tool names.
# OpenAI/DeepSeek require function names to match ^[a-zA-Z0-9_-]+$.
_DIACRITIC_MAP: dict[str, str] = {
    # Esperanto
    "ĉ": "c", "Ĉ": "C",
    "ĝ": "g", "Ĝ": "G",
    "ĥ": "h", "Ĥ": "H",
    "ĵ": "j", "Ĵ": "J",
    "ŝ": "s", "Ŝ": "S",
    "ŭ": "u", "Ŭ": "U",
    # Common European diacritics
    "à": "a", "á": "a", "â": "a", "ã": "a", "ä": "a", "å": "a",
    "è": "e", "é": "e", "ê": "e", "ë": "e",
    "ì": "i", "í": "i", "î": "i", "ï": "i",
    "ò": "o", "ó": "o", "ô": "o", "õ": "o", "ö": "o",
    "ù": "u", "ú": "u", "û": "u", "ü": "u",
    "ý": "y", "ÿ": "y",
    "À": "A", "Á": "A", "Â": "A", "Ã": "A", "Ä": "A", "Å": "A",
    "È": "E", "É": "E", "Ê": "E", "Ë": "E",
    "Ì": "I", "Í": "I", "Î": "I", "Ï": "I",
    "Ò": "O", "Ó": "O", "Ô": "O", "Õ": "O", "Ö": "O",
    "Ù": "U", "Ú": "U", "Û": "U", "Ü": "U",
    "Ý": "Y",
    "ç": "c", "Ç": "C",
    "ñ": "n", "Ñ": "N",
}

# Compiled pattern matching any character not in [a-zA-Z0-9_-]
_INVALID_CHAR = re.compile(r"[^a-zA-Z0-9_-]")


def normalize_tool_name(name: str) -> str:
    """Sanitize a tool name for LLM function calling APIs.

    Strips Esperanto and common diacritics, then replaces any remaining
    non-ASCII or special characters with underscores.

    Args:
        name: The raw tool name (e.g. ``"sistemo_particio_ŝrumpi"``).

    Returns:
        Sanitized name (e.g. ``"sistemo_particio_srumpi"``).
    """
    # Step 1: Replace known diacritics
    result = "".join(_DIACRITIC_MAP.get(c, c) for c in name)
    # Step 2: Replace any remaining non-ASCII/special chars with underscore
    result = _INVALID_CHAR.sub("_", result)
    # Step 3: Collapse multiple underscores
    while "__" in result:
        result = result.replace("__", "_")
    # Step 4: Strip leading/trailing underscores
    result = result.strip("_")
    return result
