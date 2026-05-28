"""Ensure the repository root is importable so ``import src`` works in CI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
