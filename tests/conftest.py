"""Shared fixtures and test helpers for A-kunpiloto tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from A_kunpiloto.tools.registry import ToolRegistry


# Minimal tr_multi for tests (must be defined before test_app uses it)
def tr_multi(eo: str, en: str, fr: str) -> str:
    return eo


# ---------------------------------------------------------------------------
# A minimal test module app for testing tool discovery
# ---------------------------------------------------------------------------

test_app = typer.Typer(
    help=tr_multi("Testa modulo", "Test module", "Module de test"),
)


@test_app.command()
def ls(
    kategoria: str = typer.Option("", "--kategoria", "-k", help="Filter by category"),
) -> None:
    """List test items."""
    print(f"Listing items (category={kategoria})")


@test_app.command()
def vidi(
    uuid: str = typer.Argument(..., help="Item UUID"),
) -> None:
    """View a single test item."""
    print(f"Viewing item {uuid}")


@test_app.command()
def aldoni(
    nomo: str = typer.Argument(..., help="Item name"),
    kategoria: str = typer.Option("general", "--kategoria", "-k", help="Category"),
) -> None:
    """Add a new test item."""
    print(f"Added '{nomo}' (category={kategoria})")


@test_app.command()
def forigi(
    uuid: str = typer.Argument(..., help="Item UUID"),
    jes: bool = typer.Option(False, "--jes", "-y", help="Skip confirmation"),
) -> None:
    """Delete a test item."""
    print(f"Deleted {uuid} (confirm={jes})")


# A sub-typer for nested commands
sub_app = typer.Typer()
test_app.add_typer(sub_app, name="sub", help="Sub commands")


@sub_app.command("ls")
def sub_ls() -> None:
    """List sub-items."""
    print("Sub items")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    """Return a CliRunner instance."""
    return CliRunner()


@pytest.fixture
def mock_entry_points() -> list:
    """Create mock entry points for testing tool discovery."""
    ep = MagicMock()
    ep.name = "testmod"
    ep.load.return_value = test_app
    return [ep]


@pytest.fixture
def registry(mock_entry_points) -> ToolRegistry:
    """Build a ToolRegistry with the test module registered."""
    with patch(
        "importlib.metadata.entry_points",
        return_value=mock_entry_points,
    ):
        reg = ToolRegistry()
        reg.build()
        return reg


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock LLM provider."""
    from A.core.providers import LLMResponse, ToolCall

    provider = MagicMock()
    provider.name = "test-provider"
    provider.model = "test-model"
    provider.supports_tools = True

    def chat_side_effect(messages, tools=None, **kwargs):
        """Default: return a simple text response."""
        return LLMResponse(content="Test response.")

    provider.chat.side_effect = chat_side_effect
    return provider
