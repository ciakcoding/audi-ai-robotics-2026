"""Evaluate the frozen PPO policy beyond the normal hoop-crossing terminal."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .optimize_direct import controller_action
from .sac_parameter_env import SACShotParameterEnv


def _evaluate_seed_batch(
    model_path: str,
    vecnormalize_path: str,
    seeds: list[int],
    post_shot_seconds: float,
):
    shot_env = SACShotParameterEnv()
    vector_env = DummyVecEnv([lambda: shot_env])
    vector_env = VecNormalize.load(vecnormalize_path, vector_env)
    vector_env.training = False
    vector_env.norm_reward = False
    model = PPO.load(model_path, env=vector_env, device="cpu")
    env = shot_env.base
    post_steps = int(round(post_shot_seconds / 0.02))
    records = []

    for seed in seeds:
        vector_env.seed(seed)
        observation = vector_env.reset()
        residual, _ = model.predict(observation, deterministic=True)
        _, reward, done, infos = vector_env.step(residual)
        if not bool(done[0]):
            raise RuntimeError("Parameter environment must finish in one step")

        scoring_info = dict(infos[0])
        parameters = (
            shot_env.expert_parameters
            + shot_env.parameter_scales * residual[0]
        )
        terminal_time = float(env.data.time)
        post_shot_fall = False
        first_fall_time = None
        minimum_pelvis_height = float(scoring_info["pelvis_height_m"])
        maximum_abs_pitch = abs(float(scoring_info["pitch_deg"]))
        maximum_abs_roll = abs(float(scoring_info["roll_deg"]))
        recovery_info = scoring_info

        for _ in range(post_steps):
            _, _, _, _, recovery_info = env.step(
                controller_action(env, parameters)
            )
            elapsed = float(env.data.time) - terminal_time
            if recovery_info["has_fallen"]:
                post_shot_fall = True
                if first_fall_time is None:
                    first_fall_time = elapsed
            minimum_pelvis_height = min(
                minimum_pelvis_height,
                float(recovery_info["pelvis_height_m"]),
            )
            maximum_abs_pitch = max(
                maximum_abs_pitch, abs(float(recovery_info["pitch_deg"]))
            )
            maximum_abs_roll = max(
                maximum_abs_roll, abs(float(recovery_info["roll_deg"]))
            )

        records.append(
            {
                "seed": seed,
                "episode_reward": float(reward[0]),
                "success": bool(scoring_info["success"]),
                "crossing_xy_error": scoring_info["crossing_xy_error"],
                "airborne_horizontal_distance_m": float(
                    scoring_info["airborne_horizontal_distance"]
                ),
                "release_step": int(scoring_info["release_step"]),
                "touched_backboard": bool(
                    scoring_info["touched_backboard"]
                ),
                "hoop_crossing_speed_m_s": scoring_info[
                    "hoop_crossing_speed_m_s"
                ],
                "max_rim_impact_force_n": float(
                    recovery_info["max_rim_impact_force_n"]
                ),
                "max_torso_tilt_pitch_deg": float(
                    recovery_info["max_torso_tilt_pitch_deg"]
                ),
                "max_torso_tilt_roll_deg": float(
                    recovery_info["max_torso_tilt_roll_deg"]
                ),
                "max_torso_tilt_yaw_deg": float(
                    recovery_info["max_torso_tilt_yaw_deg"]
                ),
                "final_ball_to_target_distance_m": float(
                    recovery_info["ball_to_target_distance_m"]
                ),
                "fall_before_crossing": bool(scoring_info["has_fallen"]),
                "fall_during_recovery": post_shot_fall,
                "first_fall_after_crossing_s": first_fall_time,
                "minimum_recovery_pelvis_height_m": minimum_pelvis_height,
                "maximum_recovery_abs_pitch_deg": maximum_abs_pitch,
                "maximum_recovery_abs_roll_deg": maximum_abs_roll,
                "final_pelvis_height_m": float(
                    recovery_info["pelvis_height_m"]
                ),
                "final_pitch_deg": float(recovery_info["pitch_deg"]),
                "final_roll_deg": float(recovery_info["roll_deg"]),
                "final_has_fallen": bool(recovery_info["has_fallen"]),
                "post_shot_seconds": post_shot_seconds,
            }
        )

    vector_env.close()
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vecnormalize", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--post-shot-seconds", type=float, default=6.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.post_shot_seconds < 0:
        raise ValueError("post-shot-seconds cannot be negative")
    workers = max(1, min(args.workers, args.episodes))
    seeds = list(range(args.seed, args.seed + args.episodes))
    batches = [seeds[index::workers] for index in range(workers)]

    records = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _evaluate_seed_batch,
                str(args.model),
                str(args.vecnormalize),
                batch,
                args.post_shot_seconds,
            )
            for batch in batches
            if batch
        ]
        for future in as_completed(futures):
            batch_records = future.result()
            records.extend(batch_records)
            print(f"completed {len(records)}/{args.episodes}", flush=True)

    records.sort(key=lambda row: row["seed"])
    errors = np.asarray(
        [
            row["crossing_xy_error"]
            for row in records
            if row["crossing_xy_error"] is not None
        ],
        dtype=float,
    )
    first_fall_times = [
        row["first_fall_after_crossing_s"]
        for row in records
        if row["first_fall_after_crossing_s"] is not None
    ]
    crossing_speeds = [
        row["hoop_crossing_speed_m_s"]
        for row in records
        if row["hoop_crossing_speed_m_s"] is not None
    ]
    summary = {
        "episodes": len(records),
        "seed_start": args.seed,
        "post_shot_seconds": args.post_shot_seconds,
        "successes": sum(int(row["success"]) for row in records),
        "success_rate": float(np.mean([row["success"] for row in records])),
        "mean_crossing_error": float(errors.mean()),
        "mean_episode_reward": float(
            np.mean([row["episode_reward"] for row in records])
        ),
        "max_crossing_error": float(errors.max()),
        "mean_hoop_crossing_speed_m_s": float(
            np.mean(crossing_speeds)
        ),
        "max_rim_impact_force_n": max(
            row["max_rim_impact_force_n"] for row in records
        ),
        "max_torso_tilt_pitch_deg": max(
            row["max_torso_tilt_pitch_deg"] for row in records
        ),
        "max_torso_tilt_roll_deg": max(
            row["max_torso_tilt_roll_deg"] for row in records
        ),
        "max_torso_tilt_yaw_deg": max(
            row["max_torso_tilt_yaw_deg"] for row in records
        ),
        "mean_final_ball_to_target_distance_m": float(
            np.mean(
                [
                    row["final_ball_to_target_distance_m"]
                    for row in records
                ]
            )
        ),
        "falls_before_crossing": sum(
            int(row["fall_before_crossing"]) for row in records
        ),
        "falls_during_recovery": sum(
            int(row["fall_during_recovery"]) for row in records
        ),
        "falls_at_final_frame": sum(
            int(row["final_has_fallen"]) for row in records
        ),
        "minimum_recovery_pelvis_height_m": min(
            row["minimum_recovery_pelvis_height_m"] for row in records
        ),
        "mean_minimum_recovery_pelvis_height_m": float(
            np.mean(
                [
                    row["minimum_recovery_pelvis_height_m"]
                    for row in records
                ]
            )
        ),
        "maximum_recovery_abs_pitch_deg": max(
            row["maximum_recovery_abs_pitch_deg"] for row in records
        ),
        "maximum_recovery_abs_roll_deg": max(
            row["maximum_recovery_abs_roll_deg"] for row in records
        ),
        "earliest_fall_after_crossing_s": (
            min(first_fall_times) if first_fall_times else None
        ),
    }

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output / "episodes.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
