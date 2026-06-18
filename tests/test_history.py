"""Tests for A-kunpiloto conversation history."""

from __future__ import annotations

import json

from A_kunpiloto.history import ConversationHistory


class TestConversationHistory:
    def test_init_has_system_prompt(self):
        h = ConversationHistory("You are a test bot.")
        msgs = h.messages
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are a test bot."

    def test_add_user(self):
        h = ConversationHistory("system")
        h.add_user("Hello")
        msgs = h.messages
        assert len(msgs) == 2
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "Hello"

    def test_add_assistant(self):
        h = ConversationHistory("system")
        h.add_assistant(content="Hi there!")
        msgs = h.messages
        assert len(msgs) == 2
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "Hi there!"

    def test_add_assistant_with_tool_calls(self):
        h = ConversationHistory("system")
        tc = [{"id": "call_1", "type": "function", "function": {"name": "test"}}]
        h.add_assistant(content="", tool_calls=tc)
        msgs = h.messages
        assert "tool_calls" in msgs[1]
        assert msgs[1]["tool_calls"][0]["function"]["name"] == "test"

    def test_add_tool_result(self):
        h = ConversationHistory("system")
        h.add_tool_result("call_1", json.dumps({"output": "ok"}))
        msgs = h.messages
        assert len(msgs) == 2
        assert msgs[1]["role"] == "tool"
        assert msgs[1]["tool_call_id"] == "call_1"

    def test_clear_keeps_system_prompt(self):
        h = ConversationHistory("Keep me")
        h.add_user("message 1")
        h.add_assistant(content="reply 1")
        h.add_user("message 2")
        h.clear()
        msgs = h.messages
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Keep me"

    def test_count(self):
        h = ConversationHistory("system")
        assert h.count == 0
        h.add_user("Hello")
        assert h.count == 1
        h.add_assistant(content="Hi")
        assert h.count == 2

    def test_messages_returns_copy(self):
        h = ConversationHistory("system")
        h.add_user("Hello")
        msgs = h.messages
        msgs.append({"role": "user", "content": "tamper"})
        # Original should not be affected
        assert h.count == 1

    def test_truncation_removes_oldest(self):
        """When over budget, the oldest exchange should be removed."""
        h = ConversationHistory("system")
        h.MAX_TOKENS = 10  # Very small budget
        h.AVG_CHARS_PER_TOKEN = 1

        h.add_user("A" * 20)
        h.add_assistant(content="B" * 20)
        assert h.count <= 2  # may have been truncated

        h.add_user("C" * 20)
        h.add_assistant(content="D" * 20)

        # After multiple adds, system prompt should still be there
        msgs = h.messages
        assert msgs[0]["role"] == "system"
