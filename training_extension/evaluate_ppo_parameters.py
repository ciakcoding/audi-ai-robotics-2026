from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .sac_parameter_env import SACShotParameterEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vecnormalize", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=60_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="auto",
        help="SB3 inference device. Use cpu for parallel checkpoint sweeps.",
    )
    args = parser.parse_args()

    raw_env = SACShotParameterEnv()
    vector_env = DummyVecEnv([lambda: raw_env])
    vector_env = VecNormalize.load(args.vecnormalize, vector_env)
    vector_env.training = False
    vector_env.norm_reward = False
    model = PPO.load(args.model, env=vector_env, device=args.device)

    records = []
    for episode in range(args.episodes):
        vector_env.seed(args.seed + episode)
        observation = vector_env.reset()
        action, _ = model.predict(observation, deterministic=True)
        _, reward, done, infos = vector_env.step(action)
        if not bool(done[0]):
            raise RuntimeError("Parameter environment must finish in one step")
        info = infos[0]
        records.append(
            {
                "episode": episode,
                "seed": args.seed + episode,
                "reward": float(reward[0]),
                "success": bool(info["success"]),
                "crossing_xy_error": info["crossing_xy_error"],
                "hoop_crossing_speed_m_s": info[
                    "hoop_crossing_speed_m_s"
                ],
                "max_rim_impact_force_n": info[
                    "max_rim_impact_force_n"
                ],
                "max_torso_tilt_pitch_deg": info[
                    "max_torso_tilt_pitch_deg"
                ],
                "max_torso_tilt_roll_deg": info[
                    "max_torso_tilt_roll_deg"
                ],
                "max_torso_tilt_yaw_deg": info[
                    "max_torso_tilt_yaw_deg"
                ],
                "ball_to_target_distance_m": info[
                    "ball_to_target_distance_m"
                ],
                "airborne_horizontal_distance": info[
                    "airborne_horizontal_distance"
                ],
                "touched_backboard": bool(info["touched_backboard"]),
                "has_fallen": bool(info["has_fallen"]),
                "minimum_hand_to_hoop_distance": info[
                    "minimum_hand_to_hoop_distance"
                ],
                "parameter_l2": info["parameter_l2"],
                "mean_ctrl_delta": info["mean_ctrl_delta"],
                "action": action[0].astype(float).tolist(),
            }
        )

    errors = [
        row["crossing_xy_error"]
        for row in records
        if row["crossing_xy_error"] is not None
    ]
    crossing_speeds = [
        row["hoop_crossing_speed_m_s"]
        for row in records
        if row["hoop_crossing_speed_m_s"] is not None
    ]
    summary = {
        "episodes": len(records),
        "successes": sum(int(row["success"]) for row in records),
        "success_rate": float(np.mean([row["success"] for row in records])),
        "mean_crossing_error": float(np.mean(errors)) if errors else None,
        "max_crossing_error": float(np.max(errors)) if errors else None,
        "mean_hoop_crossing_speed_m_s": (
            float(np.mean(crossing_speeds)) if crossing_speeds else None
        ),
        "max_rim_impact_force_n": float(
            np.max([row["max_rim_impact_force_n"] for row in records])
        ),
        "max_torso_tilt_pitch_deg": float(
            np.max([row["max_torso_tilt_pitch_deg"] for row in records])
        ),
        "max_torso_tilt_roll_deg": float(
            np.max([row["max_torso_tilt_roll_deg"] for row in records])
        ),
        "max_torso_tilt_yaw_deg": float(
            np.max([row["max_torso_tilt_yaw_deg"] for row in records])
        ),
        "mean_ball_to_target_distance_m": float(
            np.mean(
                [row["ball_to_target_distance_m"] for row in records]
            )
        ),
        "backboard_contacts": sum(
            int(row["touched_backboard"]) for row in records
        ),
        "falls": sum(int(row["has_fallen"]) for row in records),
        "mean_reward": float(np.mean([row["reward"] for row in records])),
        "mean_parameter_l2": float(
            np.mean([row["parameter_l2"] for row in records])
        ),
        "mean_ctrl_delta": float(
            np.mean([row["mean_ctrl_delta"] for row in records])
        ),
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
