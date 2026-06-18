"""Built-in file read/write tools for A-kunpiloto.

Provides ``write_file`` and ``read_file`` tools that the LLM can use
to interact with the filesystem. Paths are validated against a
configurable allowlist (``[tools.read]`` and ``[tools.write]``) using
A-core's ``file_security`` module.

Safety flow:
  1. Resolve the path (normalise, expand ``~``, resolve symlinks).
  2. Check for path-traversal attacks.
  3. Match against allowlist patterns:
     - Match found → auto-allow (no confirmation).
     - No match → prompt user via Rich confirmation dialog.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from A.core.file_security import match_path_globs

from A_kunpiloto.tools.safety import confirm_file_access

# ---------------------------------------------------------------------------
# Default max file size (10 MB)
# ---------------------------------------------------------------------------

DEFAULT_MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI-compatible)
# ---------------------------------------------------------------------------

WRITE_FILE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": (
            "Write text content to a file on disk. "
            "Use for saving generated content, configs, scripts, or temp files. "
            "Paths in the configured allowlist bypass confirmation; "
            "others prompt the user for permission."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "Absolute path to the target file. "
                        "May use ~ for home directory."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "Text content to write to the file.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["overwrite", "append"],
                    "description": (
                        "Write mode: 'overwrite' to replace existing content, "
                        "'append' to add to the end."
                    ),
                },
            },
            "required": ["file_path", "content"],
        },
    },
}

READ_FILE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read text content from a file on disk. "
            "Use for verification, checking written output, or reading configs. "
            "Paths in the configured allowlist bypass confirmation; "
            "others prompt the user for permission."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "Absolute path to the file to read. "
                        "May use ~ for home directory."
                    ),
                },
                "start_line": {
                    "type": "integer",
                    "description": (
                        "Starting line number (1-indexed). "
                        "Use 1 to read from the beginning. Default: 1."
                    ),
                },
                "end_line": {
                    "type": "integer",
                    "description": (
                        "Ending line number (inclusive). "
                        "Maximum 500 lines per call. "
                        "Default: read all lines (capped at 500)."
                    ),
                },
            },
            "required": ["file_path"],
        },
    },
}


# ---------------------------------------------------------------------------
# Allowlist helpers
# ---------------------------------------------------------------------------


def _get_read_patterns(config: dict[str, Any]) -> list[str]:
    """Resolve the read allowlist patterns from config.

    Write-allowed directories are **always** included in read permissions
    (so the LLM can read back what it wrote).

    Args:
        config: The ``KUNPILOTO_SCHEMA`` config dict.

    Returns:
        A list of glob pattern strings.
    """
    read_patterns: list[str] = list(
        config.get("read_always_allowed_directories") or []
    )
    write_patterns: list[str] = list(
        config.get("write_always_allowed_directories") or []
    )
    # Write directories are auto-added to read permissions
    seen = set(read_patterns)
    for wp in write_patterns:
        if wp not in seen:
            read_patterns.append(wp)
            seen.add(wp)
    return read_patterns


def _get_write_patterns(config: dict[str, Any]) -> list[str]:
    """Resolve the write allowlist patterns from config.

    Args:
        config: The config dict.

    Returns:
        A list of glob pattern strings.
    """
    return list(config.get("write_always_allowed_directories") or [])


def _check_allowlist(path: Path, patterns: list[str]) -> bool:
    """Check if *path* matches any of the allowlist patterns.

    Args:
        path: The resolved file path.
        patterns: Glob patterns from config.

    Returns:
        True if the path is allowed by at least one pattern.
    """
    return match_path_globs(path, patterns)


def _get_max_file_size(config: dict[str, Any]) -> int:
    """Get the configured max file size for writes.

    Args:
        config: The config dict.

    Returns:
        Max file size in bytes.
    """
    return int(config.get("write_max_file_size", DEFAULT_MAX_FILE_SIZE))


# ---------------------------------------------------------------------------
# Content-size helpers
# ---------------------------------------------------------------------------


def _content_preview(content: str, max_chars: int = 200) -> str:
    """Return a short preview of the content.

    Args:
        content: The full content.
        max_chars: Max characters for the preview.

    Returns:
        Truncated preview with line count.
    """
    lines = content.count("\n") + 1
    preview = content[:max_chars]
    if len(content) > max_chars:
        preview += "..."
    return f"{lines} lines, {len(content)} chars\nPreview:\n{preview}"


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def handle_write_file(
    args: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Handler for the ``write_file`` built-in tool.

    Args:
        args: Parsed arguments (file_path, content, mode).
        config: The KUNPILOTO_SCHEMA config dict.

    Returns:
        Result dict with output, error, exit_code.
    """
    file_path = args.get("file_path", "")
    content = args.get("content", "")
    mode = args.get("mode", "overwrite")

    if not file_path:
        return {"output": "", "error": "file_path is required.", "exit_code": 1}
    if mode not in ("overwrite", "append"):
        return {
            "output": "",
            "error": f"Invalid mode '{mode}'. Use 'overwrite' or 'append'.",
            "exit_code": 1,
        }

    # Validate content is text (not binary)
    if not _is_text(content):
        return {
            "output": "",
            "error": (
                "write_file supports text content only. "
                "Binary content is not supported."
            ),
            "exit_code": 1,
        }

    # Check max file size
    max_size = _get_max_file_size(config)
    if len(content) > max_size:
        return {
            "output": "",
            "error": (
                f"Content too large ({len(content)} bytes). "
                f"Maximum allowed: {max_size} bytes ({max_size // 1024 // 1024} MB)."
            ),
            "exit_code": 1,
        }

    # Normalise the path (expand ~, resolve symlinks, detect traversal)
    try:
        resolved = _normalise_path(file_path)
    except ValueError as exc:
        return {"output": "", "error": str(exc), "exit_code": 1}

    # Safety gate: check allowlist
    write_patterns = _get_write_patterns(config)
    is_auto = _check_allowlist(resolved, write_patterns)
    if not is_auto:
        preview = _content_preview(content)
        if not confirm_file_access(
            resolved,
            operation="write",
            details=preview,
        ):
            return {
                "output": "Write cancelled by user.",
                "error": "",
                "exit_code": 0,
            }

    # Ensure parent directory exists
    resolved.parent.mkdir(parents=True, exist_ok=True)

    # Write (or append)
    try:
        if mode == "append":
            with open(resolved, "a", encoding="utf-8") as f:
                f.write(content)
        else:
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(content)

        size = resolved.stat().st_size
        return {
            "output": (
                f"Wrote {size} bytes to {resolved} "
                f"(mode: {mode})."
            ),
            "error": "",
            "exit_code": 0,
        }
    except OSError as exc:
        return {
            "output": "",
            "error": f"Write error: {exc}",
            "exit_code": 1,
        }


def handle_read_file(
    args: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Handler for the ``read_file`` built-in tool.

    Args:
        args: Parsed arguments (file_path, start_line, end_line).
        config: The KUNPILOTO_SCHEMA config dict.

    Returns:
        Result dict with output, error, exit_code.
    """
    file_path = args.get("file_path", "")
    start_line = int(args.get("start_line", 1))
    end_line_arg = args.get("end_line")

    if not file_path:
        return {"output": "", "error": "file_path is required.", "exit_code": 1}

    # Normalise the path (expand ~, resolve symlinks, detect traversal)
    try:
        resolved = _normalise_path(file_path)
    except ValueError as exc:
        return {"output": "", "error": str(exc), "exit_code": 1}

    # Safety gate: check read allowlist
    read_patterns = _get_read_patterns(config)
    is_auto = _check_allowlist(resolved, read_patterns)
    if not is_auto:
        if not confirm_file_access(
            resolved,
            operation="read",
            details="",
        ):
            return {
                "output": "Read cancelled by user.",
                "error": "",
                "exit_code": 0,
            }

    # Verify file exists
    if not resolved.exists():
        return {
            "output": "",
            "error": f"File not found: {resolved}",
            "exit_code": 1,
    }
    if not resolved.is_file():
        return {
            "output": "",
            "error": f"Not a regular file: {resolved}",
            "exit_code": 1,
        }

    # Read the content
    try:
        content = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {
            "output": "",
            "error": (
                f"Cannot read '{resolved}' as text. "
                "read_file only supports text files."
            ),
            "exit_code": 1,
        }
    except OSError as exc:
        return {
            "output": "",
            "error": f"Read error: {exc}",
            "exit_code": 1,
        }

    # Detect binary content (null bytes in first 8 KB)
    if "\0" in content[:8192]:
        return {
            "output": "",
            "error": (
                f"File '{resolved}' appears to contain binary data. "
                "read_file only supports text files."
            ),
            "exit_code": 1,
        }

    lines = content.splitlines()
    total_lines = len(lines)

    # Determine line range
    if end_line_arg is not None:
        end_line = int(end_line_arg)
    else:
        end_line = total_lines

    # Cap at 500 lines
    MAX_READ_LINES = 500
    if end_line - start_line + 1 > MAX_READ_LINES:
        end_line = start_line + MAX_READ_LINES - 1

    # Clamp bounds
    start_line = max(1, start_line)
    end_line = min(total_lines, end_line)

    if start_line > end_line:
        return {
            "output": "",
            "error": (
                f"Invalid line range: {start_line}–{end_line}. "
                f"File has {total_lines} lines."
            ),
            "exit_code": 1,
        }

    selected = "\n".join(lines[start_line - 1 : end_line])

    result = (
        f"File: {resolved}\n"
        f"Total lines: {total_lines}  |  "
        f"Showing lines {start_line}–{end_line}\n"
        f"───\n"
        f"{selected}"
    )
    return {"output": result, "error": "", "exit_code": 0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_text(content: str) -> bool:
    """Check if content is valid text (not binary).

    Scans for null bytes in the first 8 KB of content.

    Args:
        content: The string content to check.

    Returns:
        True if the content appears to be text.
    """
    sample = content[:8192]
    return "\0" not in sample


def _normalise_path(path: str | Path) -> Path:
    """Normalise a user-supplied path.

    Expands ``~``, resolves symlinks, and makes the path absolute.
    Unlike :func:`A.core.file_security.resolve_safe_path`, this does
    **not** check containment against allowed bases (that is done
    later via glob matching).

    Args:
        path: The raw path from user input.

    Returns:
        The resolved, absolute :class:`Path`.
    """
    return Path(path).expanduser().resolve()


def _has_glob(pattern: str) -> bool:
    """Check if a pattern contains glob characters.

    Args:
        pattern: The pattern string.

    Returns:
        True if the pattern contains ``*``, ``?``, or ``[``.
    """
    return any(c in pattern for c in "*?[")
