"""Rich interactive REPL for A-kunpiloto.

Provides a natural-language chat interface where the user types requests
and the LLM uses A-module tools to fulfill them, with safety gates.
"""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.prompt import Prompt

from A import info, error as log_error, tr_multi

from A.core.providers import LLMProvider, ToolCall

from A_kunpiloto.commands import (
    CommandDef,
    find_command,
    resolve_template,
)
from A_kunpiloto.config import load_system_prompt
from A_kunpiloto.history import ConversationHistory
from A_kunpiloto.tools.executor import execute_tool_call
from A_kunpiloto.tools.registry import ToolRegistry
from A_kunpiloto.tools.safety import confirm_write_operation
from A_kunpiloto._display import (
    display_assistant,
    display_tool_error,
    display_tool_list,
    display_tool_result,
    display_welcome,
)
from A_kunpiloto._spinner import ThinkingSpinner

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

def _help_text(custom_commands: list[CommandDef]) -> str:
    """Build the REPL help text, optionally including custom commands.

    Args:
        custom_commands: List of available custom commands.

    Returns:
        Localized help string.
    """
    custom_lines = ""
    if custom_commands:
        lines = []
        for cmd in custom_commands:
            desc = cmd.description or tr_multi(
                "(sen priskribo)", "(no description)", "(aucune description)",
            )
            lines.append(f"  /{cmd.name:<12} {desc}")
        custom_lines = "\n\n" + tr_multi(
            "Propraj komandoj:",
            "Custom commands:",
            "Commandes personnalisées :",
        ) + "\n" + "\n".join(lines)

    return tr_multi(
        f"""
Haveblaj komandoj:
  /exit, /quit  - Eliri
  /clear        - Komenci novan konversacion
  /help         - Montri ĉi tiun helpon
  /tools        - Listi ĉiujn haveblajn ilojn{custom_lines}

Simple: tajpu vian peton en natura lingvo.
""",
        f"""
Available commands:
  /exit, /quit  - Exit
  /clear        - Start a new conversation
  /help         - Show this help
  /tools        - List all available tools{custom_lines}

Simply type your request in natural language.
""",
        f"""
Commandes disponibles :
  /exit, /quit  - Quitter
  /clear        - Nouvelle conversation
  /help         - Afficher cette aide
  /tools        - Lister tous les outils{custom_lines}

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
        custom_commands: list[CommandDef] | None = None,
    ) -> None:
        """Initialize the REPL.

        Args:
            provider: LLM provider instance.
            registry: Tool registry with discovered modules.
            max_turns: Maximum tool-calling rounds per conversation turn.
            temperature: LLM temperature setting.
            custom_commands: Custom slash-commands loaded from disk.
        """
        self._provider = provider
        self._registry = registry
        self._max_turns = max_turns
        self._temperature = temperature
        self._custom_commands = custom_commands or []
        self._history = self._build_history()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the interactive REPL loop."""
        display_welcome(WELCOME)

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
            # Show thinking spinner while waiting for LLM
            self._start_thinking()

            try:
                response = self._provider.chat(
                    self._history.messages,
                    tools=self._registry.get_schemas(),
                    temperature=self._temperature,
                )
            except Exception as exc:
                self._stop_thinking()
                log_error(tr_multi(
                    f"Eraro de LLM: {exc}",
                    f"LLM error: {exc}",
                    f"Erreur LLM : {exc}",
                ))
                return

            self._stop_thinking()

            # Extract reasoning content if present
            reasoning = getattr(response, "reasoning_content", None)

            if not response.tool_calls:
                # Pure text response — display and finish
                self._history.add_assistant(
                    content=response.content,
                    reasoning_content=reasoning,
                )
                display_assistant(response.content)
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

    # ------------------------------------------------------------------
    # Thinking indicator (simple thread-based spinner)
    # ------------------------------------------------------------------

    _spinner: ThinkingSpinner | None = None

    def _start_thinking(self) -> None:
        """Show a thinking indicator in the terminal."""
        self._spinner = ThinkingSpinner()
        self._spinner.start()

    def _stop_thinking(self) -> None:
        """Stop the thinking indicator."""
        if self._spinner:
            self._spinner.stop()
            self._spinner = None

    def _handle_tool_call(self, tc: ToolCall) -> None:
        """Execute a single tool call with safety gate and result display.

        Shows a brief "calling..." indicator while executing, then displays
        the result (success in green, error in red).

        Args:
            tc: The tool call from the LLM.
        """
        name = tc.function.get("name", "")
        raw_args = tc.function.get("arguments", "{}")

        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            err = f"Invalid JSON args: {raw_args}"
            display_tool_error(name, err)
            result = {"output": "", "error": err, "exit_code": 1}
            self._history.add_tool_result(tc.id, json.dumps(result))
            return

        if not isinstance(args, dict):
            args = {"value": str(args)}

        entry = self._registry.get_entry(name)
        if entry is None:
            err = TOOL_ERROR_MISSING.format(name=name)
            display_tool_error(name, err)
            result = {"output": "", "error": err, "exit_code": 1}
            self._history.add_tool_result(tc.id, json.dumps(result))
            return

        # Show brief "running" indicator
        _console.print(f"  [dim]▶ {entry.display_path} ...[/dim]")

        # Safety gate: for write operations, ask user
        if entry.is_write:
            if not confirm_write_operation(entry, args):
                result = {
                    "output": WRITE_CANCELLED,
                    "error": "",
                    "exit_code": 0,
                }
                display_tool_result(entry, args, result)
                self._history.add_tool_result(tc.id, json.dumps(result))
                return

        # Execute
        result = execute_tool_call(entry, args)
        display_tool_result(entry, args, result)
        self._history.add_tool_result(tc.id, json.dumps(result))

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
            _console.print(_help_text(self._custom_commands))

        elif cmd in ("/tools", "/iloj"):
            display_tool_list(self._registry)

        else:
            # Try custom slash-commands
            parts = text[1:].strip().split()  # strip leading '/', split
            cmd_name = parts[0].lower() if parts else ""
            cmd_args = parts[1:] if len(parts) > 1 else []

            matched = find_command(self._custom_commands, cmd_name)
            if matched is not None:
                resolved = resolve_template(matched.template, cmd_args)
                self._history.add_user(resolved)
                self._process_turn()
                return

            _console.print(
                tr_multi(
                    f"[red]Nekonata komando: {cmd}[/red]",
                    f"[red]Unknown command: {cmd}[/red]",
                    f"[red]Commande inconnue : {cmd}[/red]",
                )
            )

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
        base_prompt = load_system_prompt()
        system = f"{base_prompt}\n\nInstalled modules: {modules}"
        return ConversationHistory(system)
