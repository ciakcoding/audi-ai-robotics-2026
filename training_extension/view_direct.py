from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco.viewer
import numpy as np

if __package__:
    from .basketball_env import BasketballResidualEnv
    from .optimize_direct import controller_action
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training_extension.basketball_env import BasketballResidualEnv
    from training_extension.optimize_direct import controller_action


HERE = Path(__file__).resolve().parent
DEFAULT_STATE = (
    HERE
    / "cem_artifacts"
    / "selected"
    / "state.json"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Show the derived baseline with no learned/optimised residual.",
    )
    parser.add_argument("--seed", type=int, default=17000)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    args = parser.parse_args()

    state = json.loads(args.state.read_text(encoding="utf-8"))
    parameters = np.asarray(state["best_parameters"], dtype=np.float64)
    if args.baseline:
        parameters[:] = 0.0
    print(
        "播放内容："
        + ("派生 baseline（无训练修正）" if args.baseline else "训练/优化后的模型")
    )
    env = BasketballResidualEnv(curriculum_radius=0.10, set_shot_only=False)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        episode = 0
        while viewer.is_running():
            obs, _ = env.reset(seed=args.seed + episode)
            terminated = truncated = False
            info = {}
            while viewer.is_running() and not (terminated or truncated):
                start = time.perf_counter()
                action = controller_action(env, parameters)
                obs, _, terminated, truncated, info = env.step(action)
                viewer.sync()
                remaining = 0.02 - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)

            if not viewer.is_running():
                break
            print(
                "球是否达到目标："
                f"{'YES' if info['success'] else 'NO'} | "
                f"穿越误差={info['crossing_xy_error'] * 100:.2f} cm | "
                f"释放距离={info['release_distance_to_hoop_xy']:.2f} m | "
                f"飞行距离={info['airborne_horizontal_distance']:.2f} m | "
                f"碰板={'YES' if info['touched_backboard'] else 'NO'}"
            )
            end_hold = time.perf_counter() + args.hold_seconds
            while viewer.is_running() and time.perf_counter() < end_hold:
                viewer.sync()
                time.sleep(0.02)
            episode += 1

    env.close()


if __name__ == "__main__":
    main()
