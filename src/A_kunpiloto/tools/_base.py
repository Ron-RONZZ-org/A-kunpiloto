"""Base types and helper functions for A-kunpiloto tool schemas.

Shared between registry (schema generation) and executor (execution).
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_type_hints

from typer.models import (
    ArgumentInfo,
    DefaultPlaceholder,
    OptionInfo,
)


@dataclass
class ToolEntry:
    """A single tool wrapping a CLI command (or a built-in handler).

    Attributes:
        name: Canonical tool name (e.g. "vorto_aldoni" or "read_result").
        module_name: Module name (e.g. "vorto") or "_builtin" for built-ins.
        display_path: Human-readable path (e.g. "vorto aldoni").
        description: Short description for the LLM.
        schema: OpenAI-compatible tool schema dict (the full function object).
        is_write: Whether this command modifies data.
        args_prefix: CLI arg prefix (e.g. ["retposto", "sendi"]).
        positional_params: Names of positional (Argument) params, in order.
        handler: Callable for built-in tools (None for CLI tools).
        option_flags: Map from Python param name to actual CLI option flag
            (e.g. ``{"from_addr": "--from", "limit": "--limo"}``).
        injected_defaults: Defaults the executor should auto-apply before
            calling the CLI. These override CLI defaults to make the command
            LLM-friendly (e.g. ``{"stdout": True}`` to avoid opening editors).
    """
    name: str
    module_name: str
    display_path: str
    description: str
    schema: dict[str, Any] = field(default_factory=dict)
    is_write: bool = False
    args_prefix: list[str] = field(default_factory=list)
    positional_params: list[str] = field(default_factory=list)
    handler: Callable[..., dict[str, Any]] | None = None
    option_flags: dict[str, str] = field(default_factory=dict)
    injected_defaults: dict[str, Any] = field(default_factory=dict)


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


# ---------------------------------------------------------------------------
# Interactive-only and force-on CLI flags
# ---------------------------------------------------------------------------
# Options whose flags match these patterns are automatically removed from
# the LLM-visible schema — they open external programs or are meaningless
# in the copilot context.
_INTERACTIVE_ONLY_FLAGS: frozenset[str] = frozenset({
    "--html",
    "--open",
    "--browser",
})

# Options that should always be forced on for LLM calls.  Removed from
# schema and auto-injected by the executor.
_LLM_FORCE_FLAGS: frozenset[str] = frozenset({
    "--stdout",
})


def _get_cli_flag(param_default: Any) -> str | None:
    """Extract the primary CLI flag from an OptionInfo descriptor.

    Picks the first long flag (``--something``) from ``param_decls``.
    Falls back to ``None`` if the param is not an OptionInfo or has no long flags.

    Args:
        param_default: The parameter's default value (may be OptionInfo).

    Returns:
        The CLI flag string (e.g. ``"--from"``), or None.
    """
    if not isinstance(param_default, OptionInfo):
        return None
    decls = param_default.param_decls
    if not decls:
        return None
    # Prefer the first long flag (starts with --) if available
    for d in decls:
        if d.startswith("--"):
            return d
    # Fall back to the first declaration
    return decls[0]


def _flag_for_negation(flag: str) -> str:
    """Derive the negation flag for a boolean option.

    Typer auto-generates ``--no-<name>`` for every boolean option.
    This strips the leading ``--`` and prepends ``--no-``.

    Args:
        flag: The positive CLI flag (e.g. ``"--legita"``).

    Returns:
        The negation flag (e.g. ``"--no-legita"``).
    """
    return f"--no-{flag[2:]}"


def _is_interactive_only_flag(flag: str) -> bool:
    """Check if a CLI flag is interactive-only (should be hidden from LLM).

    Args:
        flag: The CLI flag string (e.g. ``"--html"``).

    Returns:
        True if the flag is considered interactive-only.
    """
    return flag in _INTERACTIVE_ONLY_FLAGS


def _is_llm_force_flag(flag: str) -> bool:
    """Check if a CLI flag should be forced on for all LLM calls.

    Args:
        flag: The CLI flag string (e.g. ``"--stdout"``).

    Returns:
        True if the flag should be auto-injected.
    """
    return flag in _LLM_FORCE_FLAGS


def _injected_default_for_flag(flag: str, param_annotation: type) -> Any:
    """Return the sensible default value for a force-on flag.

    Force-on flags (``--stdout``) → ``True`` (always print to stdout).
    Interactive-only flags → ``None`` (skip injection; CLI default applies).

    Args:
        flag: The CLI flag string.
        param_annotation: The type annotation of the parameter.

    Returns:
        The value to inject, or ``None`` to skip injection.
    """
    if flag in _LLM_FORCE_FLAGS:
        if param_annotation is bool:
            return True
        return "true"  # string-typed force flags (unusual but safe)
    # Interactive-only flags: do NOT inject — CLI defaults handle them.
    # Injecting False produces --no-<flag> which Typer may not recognise
    # for parameters with short options (e.g. --html/-H).
    return None


def build_tool_schema(
    callback: Any,
    tool_name: str,
    module_name: str,
) -> tuple[dict[str, Any], list[str], dict[str, str], dict[str, Any]]:
    """Build an OpenAI-compatible tool schema from a Typer callback function.

    Also returns a mapping from Python parameter names to their actual CLI
    option flags, and defaults that the executor should auto-apply for
    LLM-friendly operation (e.g. forcing ``--stdout``).

    Args:
        callback: The command callback function.
        tool_name: The canonical tool name.
        module_name: The A-module name (for context).

    Returns:
        A tuple of (schema_dict, positional_param_names, option_flags,
        injected_defaults).
    """
    properties: dict[str, Any] = {}
    required: list[str] = []
    positional: list[str] = []
    option_flags: dict[str, str] = {}
    injected_defaults: dict[str, Any] = {}

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

        if is_positional_param(param):
            positional.append(param_name)
            required.append(param_name)
            properties[param_name] = prop_schema
        else:
            # Extract the actual CLI flag for this option
            flag = _get_cli_flag(param.default)
            if flag:
                option_flags[param_name] = flag

                # Filter or force options that are meaningless in copilot context
                if _is_interactive_only_flag(flag) or _is_llm_force_flag(flag):
                    # Remove from schema — LLM should not see these
                    default_val = _injected_default_for_flag(flag, annotation)
                    if default_val is not None:
                        injected_defaults[param_name] = default_val
                    continue

            properties[param_name] = prop_schema

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

    return schema, positional, option_flags, injected_defaults


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
