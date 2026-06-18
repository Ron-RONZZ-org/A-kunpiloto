"""A-kunpiloto — Interactive AI copilot for the A-ecosystem.

On import, the default config is auto-seeded to
``~/.config/A/kunpiloto/config.toml`` if it does not already exist.
This ensures the file read/write tools have sensible defaults
(``/tmp/A/kunpiloto/**`` for write, ``/tmp/**`` for read) without
requiring the user to run the REPL first.
"""

from A_kunpiloto import config as _config

# Auto-seed default config at install/import time.
# Idempotent — only writes if the file does not exist.
_config._seed_default_config()

from A_kunpiloto.cli import app

__all__ = ["app"]
