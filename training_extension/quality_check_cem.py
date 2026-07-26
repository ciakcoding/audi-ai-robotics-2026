"""Code, state and physical smoke checks for the stacked CEM branch."""

from __future__ import annotations

import argparse
import json
import py_compile
from pathlib import Path

import numpy as np

from .optimize_direct import PARAMETER_NAMES, evaluate


HERE = Path(__file__).resolve().parent
DEFAULT_STATE = HERE / "cem_artifacts" / "selected" / "state.json"


def run_checks(state_path: Path, smoke_episodes: int):
    python_files = [
        HERE / "optimize_direct.py",
        HERE / "validate_direct.py",
        HERE / "render_direct.py",
        HERE / "view_direct.py",
        HERE / "quality_check_cem.py",
    ]
    for source in python_files:
        py_compile.compile(str(source), doraise=True)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    parameters = np.asarray(state["best_parameters"], dtype=np.float64)
    if parameters.shape != (len(PARAMETER_NAMES),):
        raise RuntimeError(
            f"Expected {len(PARAMETER_NAMES)} parameters, "
            f"found {parameters.shape}"
        )
    if not np.all(np.isfinite(parameters)):
        raise RuntimeError("CEM state contains non-finite parameters")

    seeds = list(range(98_200, 98_200 + smoke_episodes))
    _, records = evaluate(parameters, seeds)
    successes = sum(int(record["success"]) for record in records)
    board = sum(int(record["touched_backboard"]) for record in records)
    falls = sum(int(record["has_fallen"]) for record in records)
    if board or falls:
        raise RuntimeError(
            f"Selected CEM smoke produced board={board}, falls={falls}"
        )
    errors = [
        record["crossing_xy_error"]
        if record["crossing_xy_error"] is not None
        else record["hoop_xy_error"]
        for record in records
    ]
    return {
        "status": "PASS",
        "compiled_python_files": len(python_files),
        "algorithm": "cross-entropy method trajectory optimization",
        "is_reinforcement_learning": False,
        "parameter_count": len(parameters),
        "state_iteration": state.get("iteration"),
        "smoke_episodes": smoke_episodes,
        "smoke_successes": successes,
        "smoke_mean_crossing_error": float(np.mean(errors)),
        "backboard_contacts": board,
        "falls": falls,
        "target": [2.2, 0.0, 1.2],
        "success_radius": 0.10,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--smoke-episodes", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_checks(args.state, args.smoke_episodes)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
