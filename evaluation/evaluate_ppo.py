"""Evaluate PPO with the frozen Task 1 success and stability metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.ppo_throw_env import PPOThrowEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    env = PPOThrowEnv()
    model = PPO.load(args.model, env=env, device="cpu")
    rows = []
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
        rows.append(info)
    errors = [row["landing_error_xy"] for row in rows if row["landing_error_xy"] is not None]
    result = {
        "episodes": args.episodes,
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_landing_error_xy_m": float(np.mean(errors)) if errors else None,
        "fall_rate": float(np.mean([row["has_fallen"] for row in rows])),
        "task1_commit": "7a370663cbcc1aa96438dffc9f6331d3bf4ef35c",
    }
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    main()
