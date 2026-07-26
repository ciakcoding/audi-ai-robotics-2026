"""Repository-relative contract and smoke checks for LEVEL03 artifacts."""

from __future__ import annotations

import argparse
import json
import py_compile
from pathlib import Path

import mujoco

from .evaluate_derived_baseline import evaluate


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SCENE = HERE / "scene_throw_LEVEL03_ring.xml"
TEAMMATE_BASELINE = (
    ROOT / "scripts" / "view_baselines_LEVEL03_v031!.py"
)


def run_checks(smoke_episodes: int):
    python_files = [
        HERE / "__init__.py",
        HERE / "derived_baseline.py",
        HERE / "basketball_env.py",
        HERE / "view_derived_baseline.py",
        HERE / "evaluate_derived_baseline.py",
        HERE / "quality_check.py",
    ]
    for source in python_files:
        py_compile.compile(str(source), doraise=True)

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    names = {
        "target": mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "throw_target"
        ),
        "ball": mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "throw_ball"
        ),
        "hold": mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_EQUALITY, "hold_throw_ball"
        ),
        "backboard": mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "backboard"
        ),
    }
    rim_ids = [
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"rim_{index:02d}"
        )
        for index in range(16)
    ]
    missing = [name for name, value in names.items() if value < 0]
    if missing or any(value < 0 for value in rim_ids):
        raise RuntimeError(
            f"Scene contract missing names: {missing or 'rim segment'}"
        )
    if not TEAMMATE_BASELINE.exists():
        raise RuntimeError(
            f"Teammate baseline missing: {TEAMMATE_BASELINE}"
        )

    summary, _ = evaluate(smoke_episodes, seed=98_000)
    return {
        "status": "PASS",
        "compiled_python_files": len(python_files),
        "scene_loaded": True,
        "physical_rim_segments": len(rim_ids),
        "teammate_baseline_preserved": True,
        "smoke_evaluation": summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-episodes", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_checks(args.smoke_episodes)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
