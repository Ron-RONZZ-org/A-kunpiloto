"""Tests for A-kunpiloto ResultStore and make_tool_output."""

from __future__ import annotations

import pytest

from A_kunpiloto.result_store import (
    MAX_INLINE_CHARS,
    ResultStore,
    handle_read_result,
    make_tool_output,
)


class TestResultStore:
    def test_store_and_get_summaries(self):
        store = ResultStore()
        rid = store.store("line1\nline2\nline3", "test cmd")
        assert rid == "r0000"
        summaries = store.get_summaries()
        assert len(summaries) == 1
        assert summaries[0]["id"] == "r0000"
        assert summaries[0]["description"] == "test cmd"
        assert summaries[0]["lines"] == 3
        assert summaries[0]["chars"] == 17

    def test_store_multiple_results(self):
        store = ResultStore()
        r1 = store.store("output1", "cmd1")
        r2 = store.store("output2", "cmd2")
        assert r1 == "r0000"
        assert r2 == "r0001"
        assert len(store.get_summaries()) == 2

    def test_read_lines(self):
        store = ResultStore()
        rid = store.store("a\nb\nc\nd\ne", "test")
        assert store.read_lines(rid, 1, 3) == "a\nb\nc"
        assert store.read_lines(rid, 3, 5) == "c\nd\ne"
        assert store.read_lines(rid, 1, 1) == "a"

    def test_read_lines_clamps_bounds(self):
        store = ResultStore()
        rid = store.store("x\ny\nz", "test")
        # Start too low
        assert store.read_lines(rid, 0, 2) == "x\ny"
        # End too high
        assert store.read_lines(rid, 2, 100) == "y\nz"
        # Start > end
        assert store.read_lines(rid, 5, 2) == ""

    def test_read_lines_nonexistent_raises(self):
        store = ResultStore()
        with pytest.raises(KeyError):
            store.read_lines("r9999", 1, 10)

    def test_read_lines_respects_max(self):
        store = ResultStore()
        lines = "\n".join(f"line{i}" for i in range(500))
        rid = store.store(lines, "large")
        # Reading 300 lines should be capped at MAX_READ_LINES
        from A_kunpiloto.result_store import MAX_READ_LINES
        result = store.read_lines(rid, 1, 500)
        assert len(result.splitlines()) == MAX_READ_LINES

    def test_get_context_message(self):
        store = ResultStore()
        # Empty store
        assert store.get_context_message() is None

        store.store("data", "test cmd")
        msg = store.get_context_message()
        assert msg is not None
        assert msg["role"] == "system"
        assert "r0000" in msg["content"]
        assert "test cmd" in msg["content"]

    def test_clear(self):
        store = ResultStore()
        store.store("data", "test")
        assert len(store.get_summaries()) == 1
        store.clear()
        assert len(store.get_summaries()) == 0

    def test_multiple_clears(self):
        store = ResultStore()
        store.store("a", "first")
        store.clear()
        store.store("b", "second")
        assert store.get_summaries()[0]["id"] == "r0000"


class TestMakeToolOutput:
    def test_small_output_passthrough(self):
        store = ResultStore()
        small = "Hello, world!"
        result = make_tool_output(small, store, "test")
        assert result == small
        assert len(store.get_summaries()) == 0  # Not stored

    def test_large_output_stored(self):
        store = ResultStore()
        # Each line is 50 chars so 200 lines = 10k chars > MAX_INLINE_CHARS
        large = "\n".join(f"This is a long line of text with some padding {i:04d}" for i in range(200))
        result = make_tool_output(large, store, "big cmd")
        assert result.startswith("[Saved to r0000]")
        assert "big cmd" in result
        assert "200 lines" in result
        assert "read_result" in result
        assert len(store.get_summaries()) == 1

    def test_boundary_at_max_inline(self):
        store = ResultStore()
        # Exactly MAX_INLINE_CHARS - should fit inline
        exact = "x" * MAX_INLINE_CHARS
        result = make_tool_output(exact, store, "exact")
        assert result == exact
        assert len(store.get_summaries()) == 0

    def test_one_over_max_inline(self):
        store = ResultStore()
        # One char over - should be stored
        over = "x" * (MAX_INLINE_CHARS + 1)
        result = make_tool_output(over, store, "over")
        assert result.startswith("[Saved to")
        assert len(store.get_summaries()) == 1


class TestHandleReadResult:
    def test_read_existing(self):
        store = ResultStore()
        store.store("a\nb\nc\nd\ne", "test")
        result = handle_read_result(store, {
            "result_id": "r0000",
            "start_line": 2,
            "end_line": 4,
        })
        assert result["exit_code"] == 0
        assert result["output"] == "b\nc\nd"

    def test_read_nonexistent(self):
        store = ResultStore()
        result = handle_read_result(store, {
            "result_id": "r9999",
            "start_line": 1,
            "end_line": 10,
        })
        assert result["exit_code"] == 1
        assert "not found" in result["error"]

    def test_invalid_args(self):
        store = ResultStore()
        result = handle_read_result(store, {
            "result_id": "r0000",
            "start_line": "x",
            "end_line": 10,
        })
        assert result["exit_code"] == 1
        assert "Invalid" in result["error"]
