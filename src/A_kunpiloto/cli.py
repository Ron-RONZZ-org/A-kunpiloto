"""CLI entry point for A-kunpiloto.

Provides the ``A kunpiloto`` command with subcommands for the interactive
REPL, MCP server, and provider configuration.
"""

from __future__ import annotations

import sys
from typing import Optional

import typer
from typing_extensions import Annotated

from A import info, error, tr_multi
from A.core.ai import get_provider
from A.core.providers import LLMProvider

from A_kunpiloto.config import KUNPILOTO_SCHEMA
from A_kunpiloto.session import SessionState
from A_kunpiloto.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Typer app
# ---------------------------------------------------------------------------

app = typer.Typer(
    help=tr_multi(
        "A-kunpiloto — interaga asistanto por la A-ekosistemo.",
        "A-kunpiloto — interactive assistant for the A-ecosystem.",
        "A-kunpiloto — assistant interactif pour l'écosystème A.",
    ),
    context_settings={"help_option_names": ["-h", "--help", "--helpo"]},
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_session(
    provider_type: str | None = None,
    model: str | None = None,
    max_turns: int = 15,
    temperature: float = 0.7,
) -> SessionState:
    """Build a fully initialized session.

    Loads config from TOML (if available), merges CLI overrides,
    discovers tools, and initialises the LLM provider.

    Args:
        provider_type: Override for LLM provider type.
        model: Override for model name.
        max_turns: Override for max turns.
        temperature: Override for temperature.

    Returns:
        An initialized SessionState.

    Raises:
        SystemExit: If provider initialization fails.
    """
    session = SessionState()

    # Load config
    session.config = KUNPILOTO_SCHEMA.load()
    if provider_type:
        session.config["provider"] = provider_type
    if model:
        session.config["model"] = model
    if max_turns != 15:
        session.config["max_turns"] = max_turns
    if temperature != 0.7:
        session.config["temperature"] = temperature

    # Discover tools
    session.registry = ToolRegistry()
    session.registry.build()

    if session.registry.tool_names:
        info(tr_multi(
            f"Malkovris {len(session.registry)} ilojn el "
            f"{len(session.registry.module_names)} moduloj.",
            f"Discovered {len(session.registry)} tools from "
            f"{len(session.registry.module_names)} modules.",
            f"Découvert {len(session.registry)} outils dans "
            f"{len(session.registry.module_names)} modules.",
        ))
    else:
        info(tr_multi(
            "Neniuj A-moduloj trovita. Instalu almenaŭ unu modulon.",
            "No A-modules found. Install at least one module.",
            "Aucun module A trouvé. Installez au moins un module.",
        ))

    # Initialize provider
    try:
        provider_kwargs = {}
        if session.config.get("model"):
            provider_kwargs["model"] = session.config["model"]
        session.provider = get_provider(
            session.config["provider"],
            **provider_kwargs,
        )
        session.provider_type = session.config["provider"]
    except Exception as exc:
        error(tr_multi(
            f"Eraro dum inicializo de provizanto: {exc}",
            f"Error initialising provider: {exc}",
            f"Erreur lors de l'initialisation du fournisseur : {exc}",
        ))
        raise typer.Exit(1) from exc

    return session


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@app.command()
def repl(
    provizanto: Annotated[
        Optional[str],
        typer.Option(
            "--provizanto", "-p",
            help=tr_multi(
                "LLM-provizanto (openai, deepseek, ollama, ktp.)",
                "LLM provider (openai, deepseek, ollama, etc.)",
                "Fournisseur LLM (openai, deepseek, ollama, etc.)",
            ),
        ),
    ] = None,
    modelo: Annotated[
        Optional[str],
        typer.Option(
            "--modelo", "-m",
            help=tr_multi(
                "Modelo-nomo (apriora se ne specifita)",
                "Model name (default if not specified)",
                "Nom du modèle (défaut si non spécifié)",
            ),
        ),
    ] = None,
    maks_pasoj: Annotated[
        int,
        typer.Option(
            "--maks-pasoj",
            help=tr_multi(
                "Maksimumaj konversaciaj paŝoj",
                "Max conversation steps",
                "Étapes de conversation max",
            ),
        ),
    ] = 15,
    temperaturo: Annotated[
        float,
        typer.Option(
            "--temperaturo", "-t",
            help=tr_multi(
                "Genera temperaturo (0.0 = determinisma, 2.0 = kreema)",
                "Generation temperature (0.0 = deterministic, 2.0 = creative)",
                "Température de génération (0.0 = déterministe, 2.0 = créatif)",
            ),
        ),
    ] = 0.7,
) -> None:
    """Start the interactive REPL session.

    Discovers installed A-modules, initialises the LLM provider, and
    opens a natural-language chat interface.
    """
    session = _build_session(
        provider_type=provizanto,
        model=modelo,
        max_turns=maks_pasoj,
        temperature=temperaturo,
    )

    from A_kunpiloto.repl import REPL

    r = REPL(
        provider=session.provider,
        registry=session.registry,
        max_turns=maks_pasoj,
        temperature=temperaturo,
    )

    try:
        r.run()
    except SystemExit:
        pass


@app.command()
def mcp(
    provizanto: Annotated[
        Optional[str],
        typer.Option(
            "--provizanto", "-p",
            help=tr_multi(
                "LLM-provizanto (nur por testado)",
                "LLM provider (testing only)",
                "Fournisseur LLM (test uniquement)",
            ),
        ),
    ] = None,
    modelo: Annotated[
        Optional[str],
        typer.Option(
            "--modelo", "-m",
            help=tr_multi(
                "Modelo-nomo (apriora se ne specifita)",
                "Model name (default if not specified)",
                "Nom du modèle (défaut si non spécifié)",
            ),
        ),
    ] = None,
    transporto: Annotated[
        str,
        typer.Option(
            "--transporto",
            help=tr_multi(
                "Transporta tipo (stdio, sse)",
                "Transport type (stdio, sse)",
                "Type de transport (stdio, sse)",
            ),
        ),
    ] = "stdio",
) -> None:
    """Start the MCP server.

    Exposes discovered A-module tools via the Model Context Protocol.
    Connect any MCP client (Claude Desktop, opencode, etc.) to this
    server to use A-ecosystem tools.

    **Safety note:** The MCP server has no built-in safety gate.
    Write operations (aldoni, modifi, forigi) execute without
    confirmation. Use the ``repl`` command for safe interaction.
    """
    session = _build_session(
        provider_type=provizanto,
        model=modelo,
    )

    from A_kunpiloto.mcp import run_mcp_server

    try:
        run_mcp_server(
            registry=session.registry,
            transport=transporto,
        )
    except ImportError as exc:
        error(str(exc))
        raise typer.Exit(1) from exc


# Allow running as script
if __name__ == "__main__":
    app()
