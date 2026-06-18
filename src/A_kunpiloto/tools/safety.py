"""Safety gate — classifies commands and prompts for user confirmation.

Write operations (aldoni, modifi, forigi, etc.) require explicit user
confirmation before execution. Read operations execute freely.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.box import SIMPLE as BOX_SIMPLE

from A import tr_multi

from A_kunpiloto.tools.registry import ToolEntry

_console = Console()


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

# These command name suffixes are considered write (destructive) operations
WRITE_NAMES: frozenset[str] = frozenset({
    "aldoni", "modifi", "forigi",
    "sendi", "respondi", "plusendi",
    "forgesi", "restauxrigi", "restaŭrigi", "malplenigi",
    "importi", "movi", "kreu", "krei",
})

# These are explicitly read-only
READ_NAMES: frozenset[str] = frozenset({
    "ls", "vidi", "serci", "preni",
    "sinkronigi", "malfari", "purigi",
    "elsuti", "testi",
})


def classify_command(tool_name: str) -> str:
    """Classify a tool command as write, read, or unknown.

    Inspection is based on the last segment of the tool name,
    which corresponds to the leaf command name.

    Args:
        tool_name: The full tool name (e.g. "vorto_aldoni").

    Returns:
        "write", "read", or "unknown".
    """
    last = tool_name.split("_")[-1]
    if last in WRITE_NAMES:
        return "write"
    if last in READ_NAMES:
        return "read"
    return "unknown"


# ---------------------------------------------------------------------------
# Confirmation dialog
# ---------------------------------------------------------------------------


def _build_preview_table(entry: ToolEntry, args: dict[str, Any]) -> Panel:
    """Build a Rich Panel showing a preview of the write operation.

    Args:
        entry: The ToolEntry to preview.
        args: The arguments to display.

    Returns:
        A Rich Panel ready for rendering.
    """
    table = Table(
        show_header=True,
        box=BOX_SIMPLE,
        title_style="bold red",
    )
    table.add_column(
        tr_multi("Parametro", "Parameter", "Paramètre"),
        style="cyan",
        no_wrap=True,
    )
    table.add_column(
        tr_multi("Valoro", "Value", "Valeur"),
        style="white",
    )

    for key, value in args.items():
        table.add_row(key, str(value))

    # Build description
    module = entry.module_name
    cmd = entry.display_path.replace(" ", "·")

    return Panel(
        table,
        title=f"[bold red]⚠️  {module}·{cmd}[/bold red]",
        border_style="red",
        padding=(1, 2),
    )


def confirm_write_operation(
    entry: ToolEntry,
    args: dict[str, Any],
) -> bool:
    """Show write operation preview and ask for confirmation.

    Args:
        entry: The ToolEntry being called.
        args: The arguments being passed.

    Returns:
        True if the user confirmed, False otherwise.
    """
    _console.print()
    _console.print(_build_preview_table(entry, args))

    prompt = tr_multi(
        "Ĉu daŭrigi? [y/N]: ",
        "Continue? [y/N]: ",
        "Continuer ? [y/N] : ",
    )

    try:
        response = input(prompt).strip().lower()
        return response in ("y", "yes", "jes")
    except (EOFError, KeyboardInterrupt):
        _console.print()
        return False
