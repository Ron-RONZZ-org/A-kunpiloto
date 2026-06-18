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


def execute_tool_call(
    entry: ToolEntry,
    args: dict[str, Any],
    standalone: bool = False,
) -> dict[str, Any]:
    """Execute a tool call and return the result.

    By default runs in-process via CliRunner. When *standalone* is True,
    falls back to subprocess (useful when the module can't be imported
    cleanly in the current process).

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
    if standalone:
        return _execute_via_subprocess(entry, args)
    return _execute_via_cli_runner(entry, args)


def format_args_for_cli(
    args: dict[str, Any],
    positional_params: list[str] | None = None,
) -> list[str]:
    """Convert a dict of args to CLI arguments.

    Positional params (Arguments) are placed first without ``--`` flags,
    in the order specified by ``positional_params``. Remaining params
    are formatted as ``--option value``.

    Handles booleans (--flag for True, --no-flag for False), lists, and
    nested values as JSON strings.

    Args:
        args: The arguments dict.
        positional_params: Ordered list of positional param names.

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

    # Then option-style args (skip positional params)
    for key, value in args.items():
        if key in positional_set:
            continue
        flag = f"--{key.replace('_', '-')}"

        if isinstance(value, bool):
            if value:
                cli_args.append(flag)
            else:
                cli_args.append(f"--no-{key.replace('_', '-')}")
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
        *format_args_for_cli(args, positional_params=entry.positional_params),
    ]

    try:
        result = _runner.invoke(app, cli_args)
        return {
            "output": result.stdout or "",
            "error": result.stderr or "",
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
        *format_args_for_cli(args, positional_params=entry.positional_params),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return {
            "output": result.stdout,
            "error": result.stderr,
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
