"""Conversation history management for A-kunpiloto.

Maintains an OpenAI-format message array with automatic truncation
to stay within token budget.
"""

from __future__ import annotations

import json
from copy import deepcopy


class ConversationHistory:
    """Manages the LLM message array with token budget awareness.

    Messages follow the OpenAI chat format:
      [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "...", "tool_calls": [...]},
        {"role": "tool", "tool_call_id": "...", "content": "..."},
      ]
    """

    # Conservative token budget for tool-calling models
    MAX_TOKENS: int = 32000
    AVG_CHARS_PER_TOKEN: int = 4

    def __init__(self, system_prompt: str) -> None:
        """Initialize with a system prompt.

        Args:
            system_prompt: The system message content.
        """
        self._messages: list[dict] = [
            {"role": "system", "content": system_prompt},
        ]

    @property
    def messages(self) -> list[dict]:
        """Get a copy of the current message list."""
        return deepcopy(self._messages)

    @property
    def count(self) -> int:
        """Number of messages (excluding system prompt)."""
        return len(self._messages) - 1

    def add_user(self, content: str) -> None:
        """Add a user message and auto-truncate.

        Args:
            content: The user's message text.
        """
        self._messages.append({"role": "user", "content": content})
        self._auto_truncate()

    def add_assistant(
        self,
        content: str = "",
        tool_calls: list[dict] | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        """Add an assistant message.

        Args:
            content: Text response from the assistant.
            tool_calls: Optional list of tool call dicts (OpenAI format).
            reasoning_content: Optional reasoning/thinking content.
        """
        msg: dict = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if reasoning_content:
            msg["reasoning_content"] = reasoning_content
        self._messages.append(msg)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """Add a tool result message and auto-truncate.

        Args:
            tool_call_id: ID of the tool call this result belongs to.
            content: JSON-serialized result string.
        """
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })
        self._auto_truncate()

    def clear(self) -> None:
        """Keep only the system prompt, resetting the conversation."""
        system = self._messages[0] if self._messages else {
            "role": "system", "content": "",
        }
        self._messages = [system]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_tokens(self, messages: list[dict]) -> int:
        """Rough token count (chars / 4).

        Args:
            messages: List of message dicts.

        Returns:
            Estimated token count.
        """
        total = 0
        for m in messages:
            total += len(m.get("content", "")) // self.AVG_CHARS_PER_TOKEN
            if "tool_calls" in m:
                for tc in m["tool_calls"]:
                    fn_str = json.dumps(tc.get("function", {}))
                    total += len(fn_str) // self.AVG_CHARS_PER_TOKEN
        return total

    def _auto_truncate(self) -> None:
        """Remove oldest user+assistant pairs until under token budget."""
        while self._estimate_tokens(self._messages) > self.MAX_TOKENS:
            removed = False
            for i in range(1, len(self._messages) - 1):
                if self._messages[i]["role"] in ("user", "assistant"):
                    end = i + 1
                    while end < len(self._messages) and \
                          self._messages[end]["role"] == "tool":
                        end += 1
                    self._messages[i:end] = []
                    removed = True
                    break
            if not removed:
                break
