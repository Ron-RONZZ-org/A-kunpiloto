"""Display utilities for A-kunpiloto REPL — Rich panels and tables."""

from __future__ import annotations

from typing import Any

from rich.box import SIMPLE as BOX_SIMPLE
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from A import tr_multi

from A_kunpiloto.tools._base import ToolEntry

_console = Console()


def display_assistant(content: str) -> None:
    """Show the assistant's response in a Rich panel with Markdown rendering.

    Uses ``rich.markdown.Markdown`` to render bold, code blocks, lists,
    headings, etc. so that LLM output appears formatted in the terminal.

    Args:
        content: The text to display (raw Markdown).
    """
    rendered: Markdown | str
    if content:
        rendered = Markdown(content)
    else:
        rendered = "[dim]⋯[/dim]"

    panel = Panel(
        rendered,
        title=tr_multi(
            "[bold yellow]Kunpiloto[/bold yellow]",
            "[bold yellow]Copilot[/bold yellow]",
            "[bold yellow]Copilote[/bold yellow]",
        ),
        border_style="yellow",
        padding=(1, 2),
    )
    _console.print(panel)


def display_tool_result(
    entry: ToolEntry,
    args: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Display a tool execution result to the user.

    Success results get a green panel, errors get a red panel.

    Args:
        entry: The ToolEntry that was executed.
        args: The arguments passed.
        result: The execution result dict.
    """
    exit_code = result.get("exit_code", 0)
    output = result.get("output", "")
    error_text = result.get("error", "")

    if exit_code == 0 and not error_text:
        content = output[:2000] if output else "(ok)"
        panel = Panel(
            content,
            title=f"[bold green]✓ {entry.display_path}[/bold green]",
            border_style="green",
            padding=(0, 1),
        )
        _console.print(panel)
    else:
        lines = []
        if error_text:
            lines.append(f"[red]Error: {error_text[:500]}[/red]")
        if output:
            lines.append(output[:1000])
        content = "\n".join(lines) if lines else "(unknown error)"
        panel = Panel(
            content,
            title=f"[bold red]✗ {entry.display_path} (exit {exit_code})[/bold red]",
            border_style="red",
            padding=(0, 1),
        )
        _console.print(panel)


def display_tool_error(name: str, message: str) -> None:
    """Display a tool error that occurred before execution.

    Args:
        name: The tool name.
        message: The error message.
    """
    panel = Panel(
        f"[red]{message}[/red]",
        title=f"[bold red]✗ {name}[/bold red]",
        border_style="red",
        padding=(0, 1),
    )
    _console.print(panel)


def display_tool_list(registry: Any) -> None:
    """Display all registered tools grouped by module.

    Args:
        registry: The ToolRegistry instance.
    """
    if not registry.tool_names:
        _console.print(
            tr_multi(
                "[yellow]Neniuj iloj trovita.[/yellow]",
                "[yellow]No tools found.[/yellow]",
                "[yellow]Aucun outil trouvé.[/yellow]",
            )
        )
        return

    for mod in registry.module_names:
        mod_tools = [
            n for n in registry.tool_names
            if n.startswith(f"{mod}_") or n == mod
        ]
        if not mod_tools:
            continue

        table = Table(
            show_header=True,
            box=BOX_SIMPLE,
            title=f"[bold]{mod}[/bold]",
            title_style="bold cyan",
        )
        table.add_column(
            tr_multi("Ilo", "Tool", "Outil"),
            style="green",
            no_wrap=True,
        )
        table.add_column(
            tr_multi("Priskribo", "Description", "Description"),
            style="white",
        )
        table.add_column(
            tr_multi("Tipo", "Type", "Type"),
            style="yellow",
            width=6,
        )

        for name in sorted(mod_tools):
            entry = registry.get_entry(name)
            if entry:
                tool_type = "✏️" if entry.is_write else "📖"
                table.add_row(
                    entry.display_path,
                    entry.description[:80],
                    tool_type,
                )
        _console.print(table)
        _console.print()


def display_welcome(welcome_text: str) -> None:
    """Show the welcome message.

    Args:
        welcome_text: The welcome text (trilingual).
    """
    _console.print(welcome_text, style="bold green")
