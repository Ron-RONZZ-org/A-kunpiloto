"""Simple thread-based thinking spinner for terminal feedback."""

from __future__ import annotations

import sys
import threading
import time


class ThinkingSpinner:
    """A lightweight spinner that runs in a background thread.

    Usage::

        spinner = ThinkingSpinner()
        spinner.start()
        # ... long operation ...
        spinner.stop()
    """

    def __init__(self, message: str = "thinking...") -> None:
        """Initialise the spinner.

        Args:
            message: The text to display next to the spinner.
        """
        self._message = message
        self._active = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the spinner in a daemon thread."""
        self._active = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the spinner and clear the line."""
        self._active = False
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None
        # Clear the line
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def _spin(self) -> None:
        """Rotate the spinner characters."""
        chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while self._active:
            sys.stdout.write(f"\r  {chars[i % len(chars)]} {self._message}")
            sys.stdout.flush()
            i += 1
            time.sleep(0.1)
