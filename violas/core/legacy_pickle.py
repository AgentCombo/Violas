"""Compatibility helpers for loading legacy benchmark caches."""

from __future__ import annotations

import importlib
import sys


def enable_legacy_storage_pickle() -> None:
    aliases = {
        "storage": "violas.storage",
        "storage.vectormap": "violas.storage.vectormap",
        "storage.vectorgroup": "violas.storage.vectorgroup",
        "storage.utils": "violas.storage.utils",
    }
    for legacy_name, target_name in aliases.items():
        if legacy_name not in sys.modules:
            sys.modules[legacy_name] = importlib.import_module(target_name)
