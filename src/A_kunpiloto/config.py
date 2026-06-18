"""Configuration schema for A-kunpiloto.

Uses A-core's ConfigSchema to derive CLI options + TOML persistence
from a single declarative schema.

Users can customise the system prompt by placing a file at::

    ~/.config/A/kunpiloto/system_prompt.md

On first run (when no file exists) a shipped default is automatically
copied to that location so the user can edit it.

Custom slash-commands live in::

    ~/.config/A/kunpiloto/commands/*.md
"""

from __future__ import annotations

import importlib.resources as ilr
from pathlib import Path

from A import error as log_error
from A.core.config import ConfigSchema
from A.core.paths import config_dir

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_FILENAME = "system_prompt.md"
COMMANDS_DIRNAME = "commands"
CONFIG_FILENAME = "config.toml"

# Names of the shipped default files (package data inside ``A_kunpiloto``).
_SHIPPED_SYSTEM_PROMPT = "system_prompt.md"
_SHIPPED_CONFIG_DEFAULTS = "config.default.toml"


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


def config_path() -> Path:
    """Return the path to the user-modifiable config file.

    Returns:
        ``~/.config/A/kunpiloto/config.toml``.
    """
    return _kunpiloto_dir() / CONFIG_FILENAME


def _read_shipped(filename: str) -> str | None:
    """Read a shipped package-data file from the ``A_kunpiloto`` package.

    Args:
        filename: Name of the file (e.g. ``"system_prompt.md"``).

    Returns:
        The file contents, or ``None`` if the file cannot be read.
    """
    try:
        ref = ilr.files("A_kunpiloto").joinpath(filename)
        return ref.read_text(encoding="utf-8")
    except (FileNotFoundError, TypeError, ModuleNotFoundError, OSError) as exc:
        log_error(
            f"Ne povis legi enpakitan {filename}: {exc}",
        )
        return None


def _seed_default_prompt() -> str | None:
    """Copy the shipped system prompt to the config dir if it doesn't exist.

    Returns the prompt content (from the shipped file) or ``None`` if
    neither the shipped file nor the fallback is available.

    The config directory is created if necessary.
    """
    shipped = _read_shipped(_SHIPPED_SYSTEM_PROMPT)
    if shipped is None:
        return None

    path = system_prompt_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(shipped, encoding="utf-8")
    except OSError:
        pass  # Non-critical — we still return the content from memory
    return shipped


def load_system_prompt() -> str:
    """Load the system prompt, auto-seeding the shipped default on first run.

    Resolution order:

    1. If ``~/.config/A/kunpiloto/system_prompt.md`` exists and is
       non-empty → return its content.
    2. Otherwise, copy the shipped ``system_prompt.md`` (bundled with
       the package) to that location and return its content.
    3. Fall back to :data:`DEFAULT_SYSTEM_PROMPT`.

    Returns:
        The system prompt string.
    """
    # 1. User-customised file
    path = system_prompt_path()
    if path.exists():
        content = path.read_text(encoding="utf-8").strip()
        if content:
            return content

    # 2. Shipped default → auto-seed on first run
    seeded = _seed_default_prompt()
    if seeded is not None:
        return seeded.strip()

    # 3. Hardcoded fallback
    return DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Config file auto-seeding
# ---------------------------------------------------------------------------


def _seed_default_config() -> Path | None:
    """Copy the shipped default config to the config dir if it doesn't exist.

    Creates the config directory if necessary.  Idempotent — won't
    overwrite an existing config file the user may have edited.

    Returns:
        The config path if seeded (or already exists), ``None`` if the
        shipped file could not be read.
    """
    shipped = _read_shipped(_SHIPPED_CONFIG_DEFAULTS)
    if shipped is None:
        return None

    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        try:
            path.write_text(shipped, encoding="utf-8")
        except OSError:
            return None
    return path


def ensure_config() -> Path | None:
    """Ensure the kunpiloto config file exists, auto-seeding if needed.

    This is the public entry point (called from ``cli._build_session``).
    It ensures that both the system prompt and the default config are
    seeded on first run.

    Returns:
        The config path if available, ``None`` if seeding failed.
    """
    # 1. Ensure system prompt is seeded (existing behaviour)
    load_system_prompt()

    # 2. Ensure default config is seeded
    return _seed_default_config()


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
    # File tool configuration
    "write_always_allowed_directories": {
        "type": "str",
        "default": [],
        "help": (
            "Allowlistaj dosierujoj por write_file (globo-ŝablonoj, "
            "ekz. /tmp/A/**, ~/Documents/A/*)"
        ),
    },
    "write_max_file_size": {
        "type": "int",
        "default": 10_485_760,  # 10 MB
        "help": "Maksimuma dosiergrandeco por write_file (bajtoj)",
    },
    "read_always_allowed_directories": {
        "type": "str",
        "default": [],
        "help": (
            "Allowlistaj dosierujoj por read_file (globo-ŝablonoj). "
            "Skrib-allowlistaj dosierujoj estas aŭtomate aldonitaj."
        ),
    },
})

DEFAULT_SYSTEM_PROMPT = (
    "You are A-kunpiloto, an AI copilot for the A-ecosystem — a modular "
    "CLI framework for users who prefer the terminal over graphical "
    "interfaces. You have access to tools that wrap every A-module "
    "command, covering knowledge management, email, calendaring, system "
    "administration, media handling, and more.\n\n"
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
    "4. Be concise. Use bullet points and tables where appropriate.\n"
    "5. Respond in the language of the user request. If uncertain, default to Esperanto.\n\n"
    "ERROR RECOVERY:\n"
    "- If a tool call fails with 'not found', check the name carefully.\n"
    "  The tool name uses underscores, e.g. semantika_serci.\n"
    "- If you get a 'Saved to result_XXXX' message in tool output, "
    "use the read_result tool to read specific line ranges.\n"
    "- After 2 consecutive tool failures, stop and explain to the user "
    "what went wrong rather than trying random tools."
)
