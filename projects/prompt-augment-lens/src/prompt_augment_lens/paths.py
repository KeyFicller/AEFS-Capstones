"""Resolve package and project paths."""

from __future__ import annotations

from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PKG_DIR.parents[1]
FIXTURES_DIR = PROJECT_DIR / "fixtures"
STATIC_DIR = PKG_DIR / "static"
DEFAULT_CAPTURES_DIR = PROJECT_DIR / "captures"
