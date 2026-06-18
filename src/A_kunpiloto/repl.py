"""Rich interactive REPL for A-kunpiloto.

Provides a natural-language chat interface where the user types requests
and the LLM uses A-module tools to fulfill them, with safety gates.
"""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from A import info, error as log_error, tr_multi

from A.core.providers import LLMProvider, LLMResponse, ToolCall

from A_kunpiloto.config import DEFAULT_SYSTEM_PROMPT
from A_kunpiloto.history import ConversationHistory
from A_kunpiloto.session import SessionState
from A_kunpiloto.tools.executor import execute_tool_call
from A_kunpiloto.tools.registry import ToolRegistry
from A_kunpiloto.tools.safety import confirm_write_operation

_console = Console()


# ---------------------------------------------------------------------------
# Welcome message
# ---------------------------------------------------------------------------

WELCOME = tr_multi(
    """
╔══════════════════════════════════════╗
║  A-kunpiloto — Interaga Asistanto   ║
║                                      ║
║  Parolu nature por administri vian   ║
║  A-ekosistemon per AI.               ║
║                                      ║
║  /exit  - fini                       ║
║  /clear - nova konversacio           ║
║  /help  - helpo                      ║
║  /tools - listi haveblajn ilojn     ║
╚══════════════════════════════════════╝
""",
    """
╔══════════════════════════════════════╗
║  A-kunpiloto — Interactive Copilot  ║
║                                      ║
║  Speak naturally to manage your     ║
║  A-ecosystem with AI assistance.    ║
║                                      ║
║  /exit  - quit                       ║
║  /clear - new conversation           ║
║  /help  - help                       ║
║  /tools - list available tools       ║
╚══════════════════════════════════════╝
""",
    """
╔══════════════════════════════════════╗
║  A-kunpiloto — Copilote Interactif  ║
║                                      ║
║  Parlez naturellement pour gérer    ║
║  votre écosystème A avec IA.        ║
║                                      ║
║  /exit  - quitter                    ║
║  /clear - nouvelle conversation      ║
║  /help  - aide                       ║
║  /tools - lister les outils          ║
╚══════════════════════════════════════╝
""",
)

HELP_TEXT = tr_multi(
    """
Haveblaj komandoj:
  /exit, /quit  - Eliri
  /clear        - Komenci novan konversacion
  /help         - Montri ĉi tiun helpon
  /tools        - Listi ĉiujn haveblajn ilojn

Simple: tajpu vian peton en natura lingvo.
""",
    """
Available commands:
  /exit, /quit  - Exit
  /clear        - Start a new conversation
  /help         - Show this help
  /tools        - List all available tools

Simply type your request in natural language.
""",
    """
Commandes disponibles :
  /exit, /quit  - Quitter
  /clear        - Nouvelle conversation
  /help         - Afficher cette aide
  /tools        - Lister tous les outils

Tapez simplement votre demande en langage naturel.
""",
)

WRITE_CANCELLED = tr_multi(
    "Operacio nuligita de uzanto.",
    "Operation cancelled by user.",
    "Opération annulée par l'utilisateur.",
)

TOOL_ERROR_MISSING = tr_multi(
    "Ilo '{name}' ne trovita.",
    "Tool '{name}' not found.",
    "Outil '{name}' non trouvé.",
)

MAX_TURNS_MSG = tr_multi(
    "Pardonu, atingis maksimuman nombron da paŝoj ({n}).",
    "Sorry, reached maximum number of steps ({n}).",
    "Désolé, nombre maximum d'étapes atteint ({n}).",
)


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------


class REPL:
    """Interactive Rich-based REPL for natural language interaction."""

    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        max_turns: int = 15,
        temperature: float = 0.7,
    ) -> None:
        """Initialize the REPL.

        Args:
            provider: LLM provider instance.
            registry: Tool registry with discovered modules.
            max_turns: Maximum tool-calling rounds per conversation turn.
            temperature: LLM temperature setting.
        """
        self._provider = provider
        self._registry = registry
        self._max_turns = max_turns
        self._temperature = temperature
        self._history = self._build_history()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the interactive REPL loop."""
        _console.print(WELCOME, style="bold green")

        while True:
            try:
                user_input = Prompt.ask("[bold cyan]Vi[/bold cyan]")
            except (EOFError, KeyboardInterrupt):
                _console.print()
                self._say(tr_multi("Ĝis!", "Bye!", "Au revoir !"))
                break

            text = user_input.strip()
            if not text:
                continue

            # Handle slash commands
            if text.startswith("/"):
                self._handle_command(text)
                continue

            # Normal turn
            self._history.add_user(text)
            self._process_turn()

    # ------------------------------------------------------------------
    # Turn processing
    # ------------------------------------------------------------------

    def _process_turn(self) -> None:
        """Process a single user turn with possible tool rounds."""
        for turn in range(self._max_turns):
            try:
                response = self._provider.chat(
                    self._history.messages,
                    tools=self._registry.get_schemas(),
                    temperature=self._temperature,
                )
            except Exception as exc:
                log_error(tr_multi(
                    f"Eraro de LLM: {exc}",
                    f"LLM error: {exc}",
                    f"Erreur LLM : {exc}",
                ))
                return

            # Extract reasoning content if present
            reasoning = getattr(response, "reasoning_content", None)

            if not response.tool_calls:
                # Pure text response — display and finish
                self._history.add_assistant(
                    content=response.content,
                    reasoning_content=reasoning,
                )
                self._display_assistant(response.content)
                return

            # Has tool calls — add assistant message with tool calls
            raw_tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": tc.function,
                }
                for tc in response.tool_calls
            ]
            self._history.add_assistant(
                content=response.content or "",
                tool_calls=raw_tool_calls,
                reasoning_content=reasoning,
            )

            # Execute each tool call
            for tc in response.tool_calls:
                self._handle_tool_call(tc)

            # Continue: provider will see tool results in next iteration

        # Max turns exhausted
        self._history.add_assistant(
            content=MAX_TURNS_MSG.format(n=self._max_turns),
        )

    def _handle_tool_call(self, tc: ToolCall) -> None:
        """Execute a single tool call with safety gate.

        Args:
            tc: The tool call from the LLM.
        """
        name = tc.function.get("name", "")
        raw_args = tc.function.get("arguments", "{}")

        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            result = {"output": "", "error": f"Invalid JSON args: {raw_args}", "exit_code": 1}
            self._history.add_tool_result(tc.id, json.dumps(result))
            return

        if not isinstance(args, dict):
            args = {"value": str(args)}

        entry = self._registry.get_entry(name)
        if entry is None:
            result = {
                "output": "",
                "error": TOOL_ERROR_MISSING.format(name=name),
                "exit_code": 1,
            }
            self._history.add_tool_result(tc.id, json.dumps(result))
            return

        # Safety gate: for write operations, ask user
        if entry.is_write:
            if not confirm_write_operation(entry, args):
                result = {
                    "output": WRITE_CANCELLED,
                    "error": "",
                    "exit_code": 0,
                }
                self._history.add_tool_result(tc.id, json.dumps(result))
                return

        # Execute
        result = execute_tool_call(entry, args)
        self._history.add_tool_result(tc.id, json.dumps(result))

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _display_assistant(self, content: str) -> None:
        """Show the assistant's response in a Rich panel.

        Args:
            content: The text to display.
        """
        panel = Panel(
            content or "[dim]⋯[/dim]",
            title=tr_multi(
                "[bold yellow]Kunpiloto[/bold yellow]",
                "[bold yellow]Copilot[/bold yellow]",
                "[bold yellow]Copilote[/bold yellow]",
            ),
            border_style="yellow",
            padding=(1, 2),
        )
        _console.print(panel)

    def _say(self, text: str) -> None:
        """Print a plain message to the console.

        Args:
            text: The message to print.
        """
        info(text)

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _handle_command(self, text: str) -> None:
        """Handle a slash command.

        Args:
            text: The full command text (including slash).
        """
        cmd = text.lower().strip()

        if cmd in ("/exit", "/quit", "/q"):
            self._say(tr_multi("Ĝis!", "Bye!", "Au revoir !"))
            raise SystemExit(0)

        elif cmd in ("/clear", "/new"):
            self._history = self._build_history()
            _console.print(
                tr_multi(
                    "[dim]Nova konversacio.[/dim]",
                    "[dim]New conversation.[/dim]",
                    "[dim]Nouvelle conversation.[/dim]",
                )
            )

        elif cmd in ("/help", "/h"):
            _console.print(HELP_TEXT)

        elif cmd in ("/tools", "/iloj"):
            self._list_tools()

        else:
            _console.print(
                tr_multi(
                    f"[red]Nekonata komando: {cmd}[/red]",
                    f"[red]Unknown command: {cmd}[/red]",
                    f"[red]Commande inconnue : {cmd}[/red]",
                )
            )

    def _list_tools(self) -> None:
        """Display all registered tools grouped by module."""
        from rich.table import Table
        from rich.box import SIMPLE as BOX_SIMPLE

        if not self._registry.tool_names:
            _console.print(
                tr_multi(
                    "[yellow]Neniuj iloj trovita.[/yellow]",
                    "[yellow]No tools found.[/yellow]",
                    "[yellow]Aucun outil trouvé.[/yellow]",
                )
            )
            return

        for mod in self._registry.module_names:
            mod_tools = [
                n for n in self._registry.tool_names
                if n.startswith(f"{mod}_")
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
                entry = self._registry.get_entry(name)
                if entry:
                    tool_type = "✏️" if entry.is_write else "📖"
                    table.add_row(
                        entry.display_path,
                        entry.description[:80],
                        tool_type,
                    )
            _console.print(table)
            _console.print()

    # ------------------------------------------------------------------
    # History builder
    # ------------------------------------------------------------------

    def _build_history(self) -> ConversationHistory:
        """Build a fresh conversation history with system prompt.

        The system prompt lists available modules so the LLM knows
        what tools are at its disposal.

        Returns:
            A new ConversationHistory instance.
        """
        modules = ", ".join(self._registry.module_names) if self._registry.module_names else "(neniu)"
        system = (
            f"{DEFAULT_SYSTEM_PROMPT}\n\n"
            f"Installed modules: {modules}"
        )
        return ConversationHistory(system)
