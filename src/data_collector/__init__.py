"""Compatibility package for the existing folder named 'data collector'.

This module re-exports symbols from the local `data_loader.py` so notebooks
that import `data_collector` (the underscore package) can still work when the
code lives in `src/data collector/` (with a space).
"""
from __future__ import annotations

try:
    from .data_loader import *  # noqa: F401,F403
except Exception:
    # leave it silent — the top-level shim in src/data_collector/__init__.py will
    # also try to import from this folder when notebooks import data_collector
    pass
