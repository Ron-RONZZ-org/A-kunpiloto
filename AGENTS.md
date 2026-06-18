# AGENTS.md — Rules for A-kunpiloto

This file extends [workspace AGENTS.md](../AGENTS.md).

## Relationship to A-core

**A-kunpiloto depends on A-core** for:
- `A` package imports (i18n, output)
- `A.core.ai_config` — `get_configured_provider()`, shared provider config
- `A.core.providers` — `LLMProvider`, `chat()`, `ToolCall`, `LLMResponse`

All source code must import from `A`, never duplicate utilities.

## Module Purpose

A-kunpiloto provides an interactive natural-language interface to the
entire A-ecosystem. Instead of pre-configured AI tasks (like A-agento),
it auto-discovers all installed A-modules and exposes their CLI commands
as LLM-callable tools through a Rich-based REPL.

## Architecture

```
src/A_kunpiloto/
├── __init__.py          # Exports: app
├── _display.py          # Rich panel display helpers
├── _spinner.py          # Thinking indicator spinner
├── cli.py               # Typer app entry point
├── config.py            # Config schema
├── session.py           # Session state management
├── history.py           # Conversation history management
├── repl.py              # Rich interactive REPL loop
├── tools/
│   ├── __init__.py      # Re-exports
│   ├── _base.py         # ToolEntry dataclass, schema helpers
│   ├── registry.py      # Module discovery, tool schema generation
│   ├── executor.py      # Tool execution via CliRunner
│   └── safety.py        # Write/read classification, confirm dialogs
```

## Code Standards

1. Import from `A` — never duplicate utilities
2. Use `tr_multi()` for all user-facing strings
3. Use `error()` for errors, `info()` for info
4. Type hints on all public functions
5. Docstrings on all public functions
6. Tests required
7. No file over 500 lines — split by functional unit

## Testing

```bash
uv run pytest tests/ -v
```

Use `typer.testing.CliRunner` for CLI tests.
Mock provider calls — never hit real LLM APIs in tests.

## What to Avoid

- Don't duplicate A-core utilities
- Don't hardcode tool schemas — auto-generate from entry points
- Don't skip i18n (use `tr_multi()`)
- Don't use `print()` — use `A` output functions
