"""CLI entry point for A-kunpiloto.

Provides the ``A kunpiloto`` command with the interactive REPL.
"""

from __future__ import annotations

from typing import Optional

import typer
from typing_extensions import Annotated

from A import info, error, tr_multi
from A.core.ai_config import get_configured_provider

from A_kunpiloto.commands import load_commands
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

    # Initialize provider — uses shared A-core provider config
    # (respects providers configured via A-agento)
    try:
        provider_kwargs = {}
        if session.config.get("model"):
            provider_kwargs["model"] = session.config["model"]
        if session.config.get("temperature"):
            provider_kwargs["temperature"] = session.config["temperature"]

        # If user passed --provizanto, resolve that.
        # Otherwise, auto-fallback through prioritato order.
        session.provider = get_configured_provider(
            ref=provider_type,
            **provider_kwargs,
        )
        session.provider_type = session.provider.name or provider_type or "unknown"
    except ValueError as exc:
        error(tr_multi(
            f"Neniu provizanto havebla: {exc}",
            f"No provider available: {exc}",
            f"Aucun fournisseur disponible : {exc}",
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

    custom_commands = load_commands()
    if custom_commands:
        info(tr_multi(
            f"Ŝargis {len(custom_commands)} proprajn komandojn.",
            f"Loaded {len(custom_commands)} custom commands.",
            f"Chargé {len(custom_commands)} commandes personnalisées.",
        ))

    r = REPL(
        provider=session.provider,
        registry=session.registry,
        max_turns=maks_pasoj,
        temperature=temperaturo,
        custom_commands=custom_commands,
    )

    try:
        r.run()
    except SystemExit:
        pass


# Allow running as script
if __name__ == "__main__":
    app()
