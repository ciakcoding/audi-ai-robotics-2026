"""Deterministic evaluation of the non-learning scripted Task 1 baseline."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.g1_fixed_body_throw_env import G1FixedBodyThrowEnv


def load_policy_class():
    source = Path(__file__).with_name("view_baselines v031.py")
    spec = importlib.util.spec_from_file_location("baseline_policy_source", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load policy source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OptionCSwingPolicy


def evaluate(episodes: int, seed: int, output: Path) -> dict:
    policy_class = load_policy_class()
    env = G1FixedBodyThrowEnv()
    rows = []

    for episode in range(episodes):
        episode_seed = seed + episode
        obs, _ = env.reset(seed=episode_seed)
        policy = policy_class(env)
        terminated = truncated = False
        reward_sum = 0.0
        info = {}
        while not (terminated or truncated):
            action, _ = policy.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward

        rows.append(
            {
                "episode": episode,
                "seed": episode_seed,
                "success": bool(info["success"]),
                "landing_error_xy_m": info["landing_error_xy"],
                "landing_x_m": None if info["landing_pos"] is None else float(info["landing_pos"][0]),
                "landing_y_m": None if info["landing_pos"] is None else float(info["landing_pos"][1]),
                "release_time_s": info["release_time"],
                "has_fallen": bool(info["has_fallen"]),
                "torso_height_m": float(info["torso_height_m"]),
                "torso_tilt_deg": float(info["torso_tilt_deg"]),
                "reward_sum": float(reward_sum),
            }
        )

    errors = np.array([row["landing_error_xy_m"] for row in rows], dtype=float)
    summary = {
        "episodes": episodes,
        "seed_start": seed,
        "successes": int(sum(row["success"] for row in rows)),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_landing_error_xy_m": float(np.mean(errors)),
        "std_landing_error_xy_m": float(np.std(errors)),
        "p90_landing_error_xy_m": float(np.percentile(errors, 90)),
        "max_landing_error_xy_m": float(np.max(errors)),
        "success_radius_m": env.success_radius,
        "robot_actuators": int(env.model.nu),
        "fall_rate": float(np.mean([row["has_fallen"] for row in rows])),
        "max_torso_tilt_deg": float(max(row["torso_tilt_deg"] for row in rows)),
        "min_torso_height_m": float(min(row["torso_height_m"] for row in rows)),
        "stability_evaluable": True,
    }

    output.mkdir(parents=True, exist_ok=True)
    with (output / "baseline_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    evaluate(args.episodes, args.seed, args.output)


if __name__ == "__main__":
    main()
