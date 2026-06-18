"""Session state management for A-kunpiloto.

Holds the current provider, tool registry, conversation history,
and configuration.
"""

from __future__ import annotations

from typing import Any

from A.core.providers import LLMProvider

from A_kunpiloto.tools.registry import ToolRegistry


class SessionState:
    """Holds runtime state for a single A-kunpiloto session.

    Attributes:
        provider: The active LLM provider instance.
        provider_type: Provider type name (e.g. "openai", "deepseek").
        registry: Tool registry (auto-discovered from A.commands).
        history: ConversationHistory instance.
        config: Runtime config dict (merged CLI + TOML defaults).
    """

    def __init__(self) -> None:
        self.provider: LLMProvider | None = None
        self.provider_type: str = ""
        self.registry: ToolRegistry | None = None
        self.history: Any | None = None
        self.config: dict[str, Any] = {}

    @property
    def is_ready(self) -> bool:
        """Check if the session is ready to start the REPL."""
        return (
            self.provider is not None
            and self.registry is not None
            and self.history is not None
        )
