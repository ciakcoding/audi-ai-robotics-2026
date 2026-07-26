"""Deterministic evaluation for the derived baseline, without CEM or RL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .basketball_env import BasketballResidualEnv


def evaluate(episodes: int, seed: int):
    env = BasketballResidualEnv(curriculum_radius=0.10)
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    rows = []
    try:
        for episode in range(episodes):
            env.reset(seed=seed + episode)
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                _, _, terminated, truncated, info = env.step(zero_action)
            rows.append(
                {
                    "episode": episode,
                    "seed": seed + episode,
                    "success": bool(info["success"]),
                    "crossed_hoop_plane": bool(
                        info["crossed_hoop_plane"]
                    ),
                    "crossing_xy_error": info["crossing_xy_error"],
                    "touched_backboard": bool(
                        info["touched_backboard"]
                    ),
                    "has_fallen": bool(info["has_fallen"]),
                    "airborne_horizontal_distance": float(
                        info["airborne_horizontal_distance"]
                    ),
                    "minimum_hand_to_hoop_distance": float(
                        info["minimum_hand_to_hoop_distance"]
                    ),
                }
            )
    finally:
        env.close()

    errors = [
        row["crossing_xy_error"]
        for row in rows
        if row["crossing_xy_error"] is not None
    ]
    summary = {
        "label": "derived baseline (no CEM, no RL)",
        "episodes": episodes,
        "successes": sum(int(row["success"]) for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_crossing_error": (
            float(np.mean(errors)) if errors else None
        ),
        "max_crossing_error": float(np.max(errors)) if errors else None,
        "backboard_contacts": sum(
            int(row["touched_backboard"]) for row in rows
        ),
        "falls": sum(int(row["has_fallen"]) for row in rows),
        "target": BasketballResidualEnv.target.tolist(),
        "success_radius": BasketballResidualEnv.hoop_radius,
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary, rows = evaluate(args.episodes, args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (args.output / "episodes.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
