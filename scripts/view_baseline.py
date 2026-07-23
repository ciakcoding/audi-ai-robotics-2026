"""Portable launcher for the Task 1 baseline viewer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


try:
    import mujoco  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency {exc.name!r}. Run: "
        f"{sys.executable} -m pip install -r requirements.txt"
    ) from exc


SOURCE = Path(__file__).with_name("view_baselines v031.py")
spec = importlib.util.spec_from_file_location("teammate_baseline_viewer", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load baseline viewer: {SOURCE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.view_baseline()
