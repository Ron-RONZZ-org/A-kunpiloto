"""In-memory result store for large tool outputs.

When a tool produces output exceeding MAX_INLINE_CHARS, the output is
stored here and a reference (result_id) is returned to the LLM instead.
The LLM can then use the ``read_result`` built-in tool to read specific
line ranges on demand.
"""

from __future__ import annotations

import json
from typing import Any


# If a tool's output exceeds this many characters, it is stored
# in the ResultStore rather than inlined in the conversation.
MAX_INLINE_CHARS: int = 5000

# Maximum lines a single read_result call can return.
MAX_READ_LINES: int = 200


class ResultStore:
    """Holds oversized tool outputs for the LLM to read on demand.

    Each stored result has:
      - result_id: short hex key (e.g. "r0000")
      - output: the full output text
      - description: human-readable label (e.g. "semantika eksporti")
      - lines: number of lines
      - chars: total character count
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self._next_id: int = 0

    def store(self, output: str, description: str = "") -> str:
        """Store a tool output and return its result_id.

        Args:
            output: The full output text.
            description: Human-readable label (command path).

        Returns:
            A short hex result_id (e.g. "r0000").
        """
        result_id = f"r{self._next_id:04x}"
        self._next_id += 1
        self._store[result_id] = {
            "output": output,
            "description": description,
            "lines": output.count("\n") + 1,
            "chars": len(output),
        }
        return result_id

    def read_lines(self, result_id: str, start: int, end: int) -> str:
        """Read a range of lines from a stored result.

        Args:
            result_id: The result ID (e.g. "r0000").
            start: 1-indexed start line.
            end: Inclusive end line.

        Returns:
            The content of those lines, or an error message string.

        Raises:
            KeyError: If result_id does not exist.
        """
        data = self._store.get(result_id)
        if data is None:
            raise KeyError(f"Result '{result_id}' not found.")
        lines = data["output"].splitlines()
        # Clamp bounds
        start = max(1, start)
        end = min(len(lines), end)
        if start > end:
            return ""
        if (end - start + 1) > MAX_READ_LINES:
            end = start + MAX_READ_LINES - 1
        return "\n".join(lines[start - 1 : end])

    def get_summaries(self) -> list[dict[str, Any]]:
        """Return metadata for all stored results.

        Returns:
            List of dicts with id, description, lines, chars.
        """
        return [
            {
                "id": rid,
                "description": data["description"],
                "lines": data["lines"],
                "chars": data["chars"],
            }
            for rid, data in self._store.items()
        ]

    def get_context_message(self) -> dict | None:
        """Build a system-context message listing available results.

        Returns:
            A message dict with role='system', or None if empty.
        """
        summaries = self.get_summaries()
        if not summaries:
            return None
        lines: list[str] = [
            "Available cached results (use read_result to read):"
        ]
        for s in summaries:
            lines.append(
                f"  - {s['id']}: {s['description']} "
                f"({s['lines']} lines, {s['chars']} chars)"
            )
        return {
            "role": "system",
            "content": "\n".join(lines),
        }

    def clear(self) -> None:
        """Remove all stored results."""
        self._store.clear()
        self._next_id = 0


# ---------------------------------------------------------------------------
# Built-in tool: read_result
# ---------------------------------------------------------------------------

# The OpenAI-compatible schema for the read_result tool.
READ_RESULT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_result",
        "description": (
            "Read specific lines from a previous tool's output that was "
            "too large to display inline. Use this when a tool result "
            "says '[Saved to result_XXXX]' in its output."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "result_id": {
                    "type": "string",
                    "description": (
                        "The result ID from a previous tool output, "
                        "e.g. 'r0000'"
                    ),
                },
                "start_line": {
                    "type": "integer",
                    "description": (
                        "Starting line number (1-indexed). "
                        "Use 1 to read from the beginning."
                    ),
                },
                "end_line": {
                    "type": "integer",
                    "description": (
                        "Ending line number (inclusive). "
                        "Maximum 200 lines per call."
                    ),
                },
            },
            "required": ["result_id", "start_line", "end_line"],
        },
    },
}


def handle_read_result(
    store: ResultStore,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Handler for the read_result built-in tool.

    Args:
        store: The ResultStore instance.
        args: Parsed arguments from the LLM.

    Returns:
        Result dict in the same format as CLI tool results.
    """
    try:
        result_id = args.get("result_id", "")
        start = int(args.get("start_line", 1))
        end = int(args.get("end_line", 10))
        content = store.read_lines(result_id, start, end)
        return {
            "output": content,
            "error": "",
            "exit_code": 0,
        }
    except KeyError as e:
        # Suggest available IDs
        summaries = store.get_summaries()
        available = ", ".join(s["id"] for s in summaries) if summaries else "(none)"
        return {
            "output": "",
            "error": f"{e} Available IDs: {available}",
            "exit_code": 1,
        }
    except (ValueError, TypeError) as e:
        return {
            "output": "",
            "error": f"Invalid arguments: {e}",
            "exit_code": 1,
        }


def make_tool_output(
    output: str,
    store: ResultStore,
    description: str = "",
) -> str:
    """Conditionally store large output and return the LLM-friendly string.

    If *output* fits within MAX_INLINE_CHARS, it is returned as-is.
    Otherwise it is stored in *store* and a reference is returned.

    Args:
        output: The raw tool stdout.
        store: The ResultStore instance.
        description: Label for the result (e.g. "semantika eksporti").

    Returns:
        The string to place in the tool result's "output" field.
    """
    if len(output) <= MAX_INLINE_CHARS:
        return output

    lines = output.count("\n") + 1
    preview = "\n".join(output.splitlines()[:5])
    rid = store.store(output, description)

    return (
        f"[Saved to {rid}] Command: {description}\n"
        f"Total: {lines} lines, {len(output)} chars\n"
        f"Preview:\n{preview}\n\n"
        f"Use the read_result tool to read specific line ranges "
        f"(e.g. read_result(result_id='{rid}', start_line=1, end_line=50))."
    )
