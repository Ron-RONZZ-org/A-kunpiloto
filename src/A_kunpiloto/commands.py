"""Custom slash-commands for the A-kunpiloto REPL.

Users define commands as Markdown files in::

    ~/.config/A/kunpiloto/commands/<name>.md

Each file may contain optional YAML-like frontmatter between ``---`` markers::

    ---
    description: Show today's events and emails
    ---

    What emails have I received yesterday? From whom?
    What events do I have on my agenda today?

When the user types ``/today`` in the REPL, the template is resolved
(with ``$ARGUMENTS``, ``$1``, ``$2``, etc.) and sent to the LLM as if
the user typed the expanded text directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from A import error as log_error
from A_kunpiloto.config import commands_dir

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CommandDef:
    """A single custom slash-command definition.

    Attributes:
        name: Command name (filename stem, e.g. ``"today"``).
        description: Short description shown in ``/help``.
        template: The prompt template body (with placeholders).
    """

    name: str
    description: str = ""
    template: str = ""


# ---------------------------------------------------------------------------
# Frontmatter parser (YAML-like, no external dependency)
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(
    r"^---\s*\n"          # opening --- line
    r"(.*?)"              # frontmatter content (non-greedy)
    r"\n---\s*\n",        # closing --- line
    re.DOTALL,
)


def _parse_frontmatter(text: str) -> dict[str, Any]:
    """Extract key-value pairs from YAML-like frontmatter.

    Handles simple ``key: value`` lines. Values may be quoted (single or
    double) or unquoted.  Leading/trailing whitespace is stripped.

    Args:
        text: The raw file content (may include frontmatter).

    Returns:
        Dict of parsed key-value pairs.
    """
    result: dict[str, Any] = {}
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return result

    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, raw_val = line.partition(":")
        key = key.strip()
        val = raw_val.strip()
        # Strip surrounding quotes
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        result[key] = val

    return result


def _strip_frontmatter(text: str) -> str:
    """Remove frontmatter block from raw file text, returning the body.

    Args:
        text: Raw file content.

    Returns:
        Body text (after frontmatter), stripped.
    """
    return _FRONTMATTER_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------


def resolve_template(template: str, args: list[str]) -> str:
    """Resolve placeholders in a command template.

    Supported placeholders:

    * ``$ARGUMENTS`` — all args joined by spaces, or empty string
    * ``$@`` — same as ``$ARGUMENTS``
    * ``$1``, ``$2``, … — positional arguments (1-indexed)
    * ``$0`` — the command name (not replaced here; left as-is)

    Literal ``$`` can be written as ``$$``.

    Args:
        template: The template text with placeholders.
        args: The list of arguments typed after the command name.

    Returns:
        Template with placeholders resolved.
    """
    # First, escape literal $$
    text = template.replace("$$", "\0DOLLAR\0")

    # $ARGUMENTS / $@ → all joined
    joined = " ".join(args)
    text = text.replace("$ARGUMENTS", joined)
    text = text.replace("$@", joined)

    # $1, $2, … → individual positional args
    for i, arg in enumerate(args, start=1):
        text = text.replace(f"${i}", arg)

    # Restore literal $
    text = text.replace("\0DOLLAR\0", "$")

    return text


# ---------------------------------------------------------------------------
# Loading commands from disk
# ---------------------------------------------------------------------------


def load_commands(custom_dir: Path | None = None) -> list[CommandDef]:
    """Load all custom commands from the commands directory.

    Args:
        custom_dir: Directory to scan for ``*.md`` files.
                    Defaults to ``~/.config/A/kunpiloto/commands/``.

    Returns:
        List of :class:`CommandDef` objects, sorted alphabetically by name.
    """
    directory = custom_dir or commands_dir()
    if not directory.is_dir():
        return []

    result: list[CommandDef] = []

    for md_path in sorted(directory.iterdir()):
        if md_path.suffix.lower() != ".md":
            continue
        try:
            raw = md_path.read_text(encoding="utf-8")
        except OSError as exc:
            log_error(f"Error reading command file {md_path}: {exc}")
            continue

        front = _parse_frontmatter(raw)
        body = _strip_frontmatter(raw)

        if not body:
            continue  # Skip empty commands

        result.append(CommandDef(
            name=md_path.stem,
            description=front.get("description", ""),
            template=body,
        ))

    return result


def find_command(commands: list[CommandDef], name: str) -> CommandDef | None:
    """Find a command by name (case-insensitive).

    Args:
        commands: List of available commands.
        name: Command name to look up (without leading ``/``).

    Returns:
        The matching :class:`CommandDef`, or ``None``.
    """
    name_lower = name.lower()
    for cmd in commands:
        if cmd.name.lower() == name_lower:
            return cmd
    return None
