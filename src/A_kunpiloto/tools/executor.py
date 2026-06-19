"""Tool execution — runs CLI commands via Typer's CliRunner or subprocess.

Uses ``typer.testing.CliRunner`` for in-process execution (faster,
better error messages) with subprocess fallback for edge cases.
"""

from __future__ import annotations

import json
import shlex
from typing import Any

import typer
from typer.testing import CliRunner

from A_kunpiloto.tools.registry import ToolEntry


_runner = CliRunner()

# Known CLI error patterns mapped to LLM-friendly messages.
_ERROR_PATTERNS: dict[str, str] = {
    "No such command": (
        "The command could not be found in module '{module}'. "
        "Check the function names in the tool list — the expected name "
        "may have a different prefix or be a subcommand of another group."
    ),
    "Missing argument": (
        "Missing required argument(s). Check the tool's parameter names "
        "and provide all required arguments."
    ),
    "No such option": (
        "Unknown option. The parameter name does not match the "
        "tool's expected options. Check the tool definition for "
        "the correct option names."
    ),
    "Error: Invalid value": (
        "Invalid value for a parameter. Check the expected types "
        "(string, integer, boolean, etc.) and try again."
    ),
}


def format_structured_error(
    entry: ToolEntry,
    stderr: str,
) -> str:
    """Convert raw CLI stderr into an LLM-friendly error message.

    Args:
        entry: The tool entry that was executed.
        stderr: The raw stderr from the CLI execution.

    Returns:
        A cleaned-up error message the LLM can understand.
    """
    if not stderr:
        return "Unknown error (exit code != 0)."

    for pattern, template in _ERROR_PATTERNS.items():
        if pattern in stderr:
            return template.format(module=entry.module_name)

    # Fallback: truncate raw error to 500 chars
    if len(stderr) > 500:
        return stderr[:500] + f"\n... (truncated, total {len(stderr)} chars)"
    return stderr


def _merge_injected_defaults(
    args: dict[str, Any],
    injected_defaults: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge injected defaults into args, preserving LLM-provided values.

    The LLM's explicit args take priority over injected defaults, but
    injected defaults fill in any values the LLM didn't specify.

    Args:
        args: The arguments from the LLM.
        injected_defaults: Defaults from the tool schema.

    Returns:
        Merged args dict.
    """
    if not injected_defaults:
        return args
    merged = dict(injected_defaults)
    merged.update(args)  # LLM values override injected defaults
    return merged


def execute_tool_call(
    entry: ToolEntry,
    args: dict[str, Any],
    standalone: bool = False,
) -> dict[str, Any]:
    """Execute a tool call and return the result.

    By default runs in-process via CliRunner. When *standalone* is True,
    falls back to subprocess (useful when the module can't be imported
    cleanly in the current process).

    Injected defaults from the tool entry (e.g. ``--stdout``) are merged
    into *args* before execution, with explicit LLM args taking priority.

    Args:
        entry: The ToolEntry to execute.
        args: The arguments dict from the LLM.
        standalone: If True, run via subprocess instead of CliRunner.

    Returns:
        A dict with keys:
            - "output": stdout content (str).
            - "error": stderr content (str, empty on success).
            - "exit_code": int (0 = success).
    """
    effective_args = _merge_injected_defaults(args, entry.injected_defaults)
    if standalone:
        return _execute_via_subprocess(entry, effective_args)
    return _execute_via_cli_runner(entry, effective_args)


def format_args_for_cli(
    args: dict[str, Any],
    positional_params: list[str] | None = None,
    option_flags: dict[str, str] | None = None,
) -> list[str]:
    """Convert a dict of args to CLI arguments.

    Positional params (Arguments) are placed first without ``--`` flags,
    in the order specified by ``positional_params``. Remaining params
    are formatted as ``--option value``.

    When *option_flags* is provided, it maps Python parameter names to
    the actual CLI flag names defined in the Typer command (e.g.
    ``{"from_addr": "--from"}``).  Without it, flags are derived from
    parameter names (``from_addr`` → ``--from-addr``).

    Handles booleans (--flag for True, --no-flag for False), lists, and
    nested values as JSON strings.

    Args:
        args: The arguments dict.
        positional_params: Ordered list of positional param names.
        option_flags: Optional map from param name to CLI flag.

    Returns:
        A list of CLI argument strings.
    """
    positional = positional_params or []
    positional_set = set(positional)
    cli_args: list[str] = []

    # Positional args first, in definition order
    for key in positional:
        if key in args:
            cli_args.append(str(args[key]))

    # Then option-style args (skip positional params and None values)
    for key, value in args.items():
        if key in positional_set or value is None:
            continue

        # Use the actual CLI flag if available, otherwise derive from param name
        if option_flags and key in option_flags:
            flag = option_flags[key]
        else:
            flag = f"--{key.replace('_', '-')}"

        if isinstance(value, bool):
            if value:
                cli_args.append(flag)
            else:
                # Typer auto-generates --no-<name> for boolean options
                cli_args.append(f"--no-{flag[2:]}")
        elif isinstance(value, list):
            cli_args.append(flag)
            cli_args.append(json.dumps(value))
        elif isinstance(value, dict):
            cli_args.append(flag)
            cli_args.append(json.dumps(value))
        else:
            cli_args.append(flag)
            cli_args.append(str(value))
    return cli_args


# ---------------------------------------------------------------------------
# In-process execution (CliRunner)
# ---------------------------------------------------------------------------


def _discover_app(module_name: str) -> typer.Typer | None:
    """Find a Typer app for a module by its entry point name.

    Args:
        module_name: The entry point name (e.g. "vorto").

    Returns:
        The Typer app, or None if not found.
    """
    import importlib.metadata
    for ep in importlib.metadata.entry_points(group="A.commands"):
        if ep.name == module_name:
            try:
                return ep.load()
            except Exception:
                return None
    return None


def _execute_via_cli_runner(
    entry: ToolEntry,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Execute a tool in-process using Typer's CliRunner.

    Args:
        entry: The ToolEntry.
        args: The arguments dict.

    Returns:
        Result dict with output, error, exit_code.
    """
    app = _discover_app(entry.module_name)
    if app is None:
        return {
            "output": "",
            "error": f"Module '{entry.module_name}' not found or failed to load.",
            "exit_code": 1,
        }

    cli_args = [
        *entry.args_prefix,
        *format_args_for_cli(
            args,
            positional_params=entry.positional_params,
            option_flags=entry.option_flags,
        ),
    ]

    try:
        result = _runner.invoke(app, cli_args)
        error = result.stderr or ""
        if result.exit_code and error:
            error = format_structured_error(entry, error)
        return {
            "output": result.stdout or "",
            "error": error,
            "exit_code": result.exit_code if result.exit_code is not None else 0,
        }
    except Exception as exc:
        return {
            "output": "",
            "error": f"Execution error: {exc}",
            "exit_code": 1,
        }


# ---------------------------------------------------------------------------
# Subprocess fallback
# ---------------------------------------------------------------------------


def _execute_via_subprocess(
    entry: ToolEntry,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Execute a tool via subprocess (standalone mode).

    Args:
        entry: The ToolEntry.
        args: The arguments dict.

    Returns:
        Result dict with output, error, exit_code.
    """
    import subprocess
    import sys

    cmd = [
        sys.executable, "-m", "A",
        *entry.args_prefix,
        *format_args_for_cli(
            args,
            positional_params=entry.positional_params,
            option_flags=entry.option_flags,
        ),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        error = result.stderr or ""
        if result.returncode and error:
            error = format_structured_error(entry, error)
        return {
            "output": result.stdout,
            "error": error,
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "output": "",
            "error": "Command timed out after 120 seconds.",
            "exit_code": 1,
        }
    except FileNotFoundError:
        return {
            "output": "",
            "error": "A CLI not found in PATH.",
            "exit_code": 1,
        }
    except Exception as exc:
        return {
            "output": "",
            "error": f"Subprocess error: {exc}",
            "exit_code": 1,
        }
