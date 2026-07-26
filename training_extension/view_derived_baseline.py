"""Interactive viewer for the derived baseline, with no CEM dependency."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

if __package__:
    from .basketball_env import BasketballResidualEnv
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training_extension.basketball_env import BasketballResidualEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17_000)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    args = parser.parse_args()

    import mujoco.viewer

    env = BasketballResidualEnv(curriculum_radius=0.10)
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    print(
        "Playback: modified baseline (no CEM, no RL)\n"
        "Target=(2.2, 0.0, 1.2), fixed ball-centre "
        "hoop-crossing radius=0.10 m"
    )
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        episode = 0
        while viewer.is_running():
            _, _ = env.reset(seed=args.seed + episode)
            terminated = truncated = False
            info = {}
            while viewer.is_running() and not (terminated or truncated):
                start = time.perf_counter()
                _, _, terminated, truncated, info = env.step(zero_action)
                viewer.sync()
                remaining = 0.02 - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)
            if not viewer.is_running():
                break
            crossing_error = info.get("crossing_xy_error")
            crossing_text = (
                "N/A"
                if crossing_error is None
                else f"{100.0 * crossing_error:.2f} cm"
            )
            print(
                "Ball reached target: "
                f"{'YES' if info['success'] else 'NO'} | "
                f"Distance from hoop centre at crossing={crossing_text} | "
                f"Backboard contact="
                f"{'YES' if info['touched_backboard'] else 'NO'} | "
                f"Robot fell={'YES' if info['has_fallen'] else 'NO'}"
            )
            end_hold = time.perf_counter() + args.hold_seconds
            while viewer.is_running() and time.perf_counter() < end_hold:
                viewer.sync()
                time.sleep(0.02)
            episode += 1
    env.close()


if __name__ == "__main__":
    main()
