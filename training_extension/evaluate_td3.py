from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .td3_residual_env import TD3BasketballResidualEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vecnormalize", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=60_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw_env = TD3BasketballResidualEnv()
    vector_env = DummyVecEnv([lambda: raw_env])
    vector_env = VecNormalize.load(args.vecnormalize, vector_env)
    vector_env.training = False
    vector_env.norm_reward = False
    model = TD3.load(args.model, env=vector_env, device="auto")

    records = []
    for episode in range(args.episodes):
        vector_env.seed(args.seed + episode)
        observation = vector_env.reset()
        done = np.asarray([False])
        episode_reward = 0.0
        final_info = {}
        while not bool(done[0]):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, infos = vector_env.step(action)
            episode_reward += float(reward[0])
            final_info = infos[0]
        records.append(
            {
                "episode": episode,
                "seed": args.seed + episode,
                "reward": episode_reward,
                "success": bool(final_info.get("success", False)),
                "crossing_xy_error": final_info.get("crossing_xy_error"),
                "airborne_horizontal_distance": final_info.get(
                    "airborne_horizontal_distance"
                ),
                "release_distance_to_hoop_xy": final_info.get(
                    "release_distance_to_hoop_xy"
                ),
                "release_pelvis_distance_to_hoop_xy": final_info.get(
                    "release_pelvis_distance_to_hoop_xy"
                ),
                "minimum_hand_to_hoop_distance": final_info.get(
                    "minimum_hand_to_hoop_distance"
                ),
                "touched_backboard": bool(
                    final_info.get("touched_backboard", False)
                ),
                "has_fallen": bool(final_info.get("has_fallen", False)),
                "rl_action_delta": final_info.get("rl_action_delta"),
                "ctrl_delta": final_info.get("ctrl_delta"),
                "foot_slip_mps": final_info.get("foot_slip_mps"),
            }
        )

    successful_errors = [
        row["crossing_xy_error"]
        for row in records
        if row["crossing_xy_error"] is not None
    ]
    summary = {
        "episodes": len(records),
        "successes": sum(int(row["success"]) for row in records),
        "success_rate": float(np.mean([row["success"] for row in records])),
        "mean_crossing_error": (
            float(np.mean(successful_errors)) if successful_errors else None
        ),
        "max_crossing_error": (
            float(np.max(successful_errors)) if successful_errors else None
        ),
        "backboard_contacts": sum(
            int(row["touched_backboard"]) for row in records
        ),
        "falls": sum(int(row["has_fallen"]) for row in records),
        "mean_reward": float(np.mean([row["reward"] for row in records])),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output / "episodes.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    vector_env.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
