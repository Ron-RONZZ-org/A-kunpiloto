"""Configuration schema for A-kunpiloto.

Uses A-core's ConfigSchema to derive CLI options + TOML persistence
from a single declarative schema.
"""

from __future__ import annotations

from A.core.config import ConfigSchema

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
    "Rules:\n"
    "1. Use tools when the user asks for actions (listing, searching, "
    "adding, modifying, deleting data).\n"
    "2. After executing tools, summarize the results for the user.\n"
    "3. If you need more information, ask the user before calling tools.\n"
    "4. Be concise but helpful.\n"
    "5. Respond in the user's language (if unspecified, use Esperanto)."
)
