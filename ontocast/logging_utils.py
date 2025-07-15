"""
Colored logging utilities for OntoCast.

Provides lightweight colored logging formatters using ANSI color codes.
"""

import logging
import os
import sys


class ColoredFormatter(logging.Formatter):
    """Lightweight colored formatter using ANSI color codes."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[1;31m",  # Bold Red
    }

    RESET = "\033[0m"  # Reset to default color

    def __init__(self, fmt=None, datefmt=None, use_colors=True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and self._supports_color()

    def _supports_color(self):
        """Check if the terminal supports color output."""
        return (
            hasattr(sys.stderr, "isatty")
            and sys.stderr.isatty()
            and "TERM" in os.environ
            and os.environ["TERM"] != "dumb"
        )

    def format(self, record):
        if self.use_colors:
            # Get the color for this log level
            color = self.COLORS.get(record.levelname, "")

            # Format the record
            formatted = super().format(record)

            # Apply color to the level name in the formatted string
            if color:
                # Replace the level name with colored version
                level_name = record.levelname
                colored_level = f"{color}{level_name}{self.RESET}"
                formatted = formatted.replace(level_name, colored_level, 1)

            return formatted
        else:
            return super().format(record)
