"""Shared project paths."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "outputs" / "videos"

