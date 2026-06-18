"""Configuration schema for A-kunpiloto.

Uses A-core's ConfigSchema to derive CLI options + TOML persistence
from a single declarative schema.

Users can customise the system prompt by placing a file at::

    ~/.config/A/kunpiloto/system_prompt.md

Custom slash-commands live in::

    ~/.config/A/kunpiloto/commands/*.md
"""

from __future__ import annotations

from pathlib import Path

from A.core.config import ConfigSchema
from A.core.paths import config_dir

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_FILENAME = "system_prompt.md"
COMMANDS_DIRNAME = "commands"


def _kunpiloto_dir() -> Path:
    """Return the A-kunpiloto config directory.

    Returns:
        ``~/.config/A/kunpiloto/`` (or ``$A_DIR/config/kunpiloto/``).
    """
    return config_dir() / "kunpiloto"


def system_prompt_path() -> Path:
    """Return the path to the user-modifiable system prompt file.

    Returns:
        ``~/.config/A/kunpiloto/system_prompt.md``.
    """
    return _kunpiloto_dir() / SYSTEM_PROMPT_FILENAME


def commands_dir() -> Path:
    """Return the path to the custom commands directory.

    Returns:
        ``~/.config/A/kunpiloto/commands/``.
    """
    return _kunpiloto_dir() / COMMANDS_DIRNAME


def load_system_prompt() -> str:
    """Load the system prompt from the user config dir, or return the default.

    If ``~/.config/A/kunpiloto/system_prompt.md`` exists, its contents
    (after stripping leading/trailing whitespace) are returned.
    Otherwise :data:`DEFAULT_SYSTEM_PROMPT` is used.

    Returns:
        The system prompt string.
    """
    path = system_prompt_path()
    if path.exists():
        content = path.read_text(encoding="utf-8").strip()
        if content:
            return content
    return DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

KUNPILOTO_SCHEMA = ConfigSchema("kunpiloto", {
    "provider": {
        "type": "str",
        "default": "openai",
        "help": "LLM-provizanto (openai, deepseek, ollama, ktp.)",
    },
    "model": {
        "type": "str",
        "default": "",
        "help": "Modelo-nomo (malplena = apriora de provizanto)",
    },
    "max_turns": {
        "type": "int",
        "default": 15,
        "help": "Maksimumaj konversaciaj paŝoj",
    },
    "temperature": {
        "type": "float",
        "default": 0.7,
        "help": "Genera temperaturo (0.0 = determinisma, 2.0 = kreema)",
    },
})

DEFAULT_SYSTEM_PROMPT = (
    "You are A-kunpiloto, an AI copilot for the A-ecosystem — a CLI-based "
    "personal knowledge management system. You have access to tools that "
    "wrap every A-module command.\n\n"
    "Your job is to understand the user's natural-language request and "
    "use the appropriate tools to fulfill it.\n\n"
    "TOOL NAMING:\n"
    "Tool names use underscores: module_command or module_subgroup_command.\n"
    "Example: semantika_serci, not 'semantika serci'.\n\n"
    "RULES:\n"
    "1. Use tools when the user asks for actions (listing, searching, "
    "adding, modifying, deleting data).\n"
    "2. After executing tools, summarize the results for the user.\n"
    "   If a tool already returned the answer you need, stop calling "
    "more tools and respond immediately.\n"
    "3. If you need more information, ask the user before calling tools.\n"
    "4. Be concise but helpful.\n"
    "5. Respond in the user's language (if unspecified, use Esperanto).\n\n"
    "ERROR RECOVERY:\n"
    "- If a tool call fails with 'not found', check the name carefully.\n"
    "  The tool name uses underscores, e.g. semantika_serci.\n"
    "- If you get a 'Saved to result_XXXX' message in tool output, "
    "use the read_result tool to read specific line ranges.\n"
    "- After 2 consecutive tool failures, stop and explain to the user "
    "what went wrong rather than trying random tools."
)
