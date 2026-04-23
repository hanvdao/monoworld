"""Thin logging wrapper so we don't sprinkle logger configs everywhere."""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "monoworld") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
        )
        _CONFIGURED = True
    return logging.getLogger(name)
