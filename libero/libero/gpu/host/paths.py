"""Canonical source locations for checkout, editable and wheel installs."""
from pathlib import Path
GPU_ROOT = Path(__file__).resolve().parents[1]
# Wheels place the canonical GPU package at libero/gpu. Source keeps the
# single implementation under libero/libero/gpu; no alias modules are created.
REPO_ROOT = GPU_ROOT.parents[2] if GPU_ROOT.parent.name == "libero" and GPU_ROOT.parent.parent.name == "libero" else GPU_ROOT.parents[1]
