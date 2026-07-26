from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

if __package__:
    from .optimize_direct import controller_action
    from .sac_parameter_env import SACShotParameterEnv
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training_extension.optimize_direct import controller_action
    from training_extension.sac_parameter_env import SACShotParameterEnv


HERE = Path(__file__).resolve().parent
DEFAULT_FROZEN = (
    HERE / "frozen" / "ppo_parameters_12288_selected_20260726"
)


def realtime_step(env, parameters, viewer):
    start = time.perf_counter()
    result = env.step(controller_action(env, parameters))
    viewer.sync()
    remaining = 0.02 - (time.perf_counter() - start)
    if remaining > 0:
        time.sleep(remaining)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_FROZEN / "selected_model.zip",
    )
    parser.add_argument(
        "--vecnormalize",
        type=Path,
        default=DEFAULT_FROZEN / "selected_vecnormalize.pkl",
    )
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--post-shot-seconds", type=float, default=10.0)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    args = parser.parse_args()

    shot_env = SACShotParameterEnv()
    vector_env = DummyVecEnv([lambda: shot_env])
    vector_env = VecNormalize.load(args.vecnormalize, vector_env)
    vector_env.training = False
    vector_env.norm_reward = False
    model = PPO.load(args.model, env=vector_env, device="cpu")
    env = shot_env.base
    print(
        "Playback: frozen PPO 12,288 parameter-residual policy "
        "(full walking motion and direct shot)"
    )

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        episode = 0
        while viewer.is_running():
            vector_env.seed(args.seed + episode)
            observation = vector_env.reset()
            residual, _ = model.predict(
                observation, deterministic=True
            )
            parameters = (
                shot_env.expert_parameters
                + shot_env.parameter_scales * residual[0]
            )
            terminated = truncated = False
            info = {}
            while viewer.is_running() and not (terminated or truncated):
                _, _, terminated, truncated, info = realtime_step(
                    env, parameters, viewer
                )
            if not viewer.is_running():
                break

            scoring_info = dict(info)
            post_shot_fall = False
            minimum_pelvis_height = float(scoring_info["pelvis_height_m"])
            recovery_end = (
                float(env.data.time) + max(args.post_shot_seconds, 0.0)
            )
            while viewer.is_running() and env.data.time < recovery_end:
                _, _, _, _, info = realtime_step(
                    env, parameters, viewer
                )
                post_shot_fall = post_shot_fall or bool(info["has_fallen"])
                minimum_pelvis_height = min(
                    minimum_pelvis_height,
                    float(info["pelvis_height_m"]),
                )
            if not viewer.is_running():
                break

            print("\n--- FINAL DISPLAY METRICS (same definitions as v031) ---")
            print(
                "Hoop-crossing speed="
                f"{scoring_info['hoop_crossing_speed_m_s']:.2f} m/s"
            )
            print(
                "Max rim impact force="
                f"{info['max_rim_impact_force_n']:.2f} N"
            )
            print(
                "Max torso tilt (pitch / roll / yaw)="
                f"{info['max_torso_tilt_pitch_deg']:.1f}° / "
                f"{info['max_torso_tilt_roll_deg']:.1f}° / "
                f"{info['max_torso_tilt_yaw_deg']:.1f}°"
            )
            print(
                "Final ball-to-target distance="
                f"{info['ball_to_target_distance_m']:.2f} m"
            )
            print("--- LEGACY RL METRICS (retained) ---")
            print(
                "Ball reached target: "
                f"{'YES' if scoring_info['success'] else 'NO'} | "
                f"Distance from hoop centre at crossing="
                f"{scoring_info['crossing_xy_error'] * 100:.2f} cm | "
                f"Airborne distance="
                f"{scoring_info['airborne_horizontal_distance']:.2f} m | "
                f"Backboard contact="
                f"{'YES' if scoring_info['touched_backboard'] else 'NO'} | "
                f"Fall before crossing="
                f"{'YES' if scoring_info['has_fallen'] else 'NO'} | "
                f"Fall during {args.post_shot_seconds:.1f} s recovery="
                f"{'YES' if post_shot_fall else 'NO'} | "
                f"Minimum pelvis height={minimum_pelvis_height:.3f} m"
            )
            end_hold = time.perf_counter() + args.hold_seconds
            while viewer.is_running() and time.perf_counter() < end_hold:
                viewer.sync()
                time.sleep(0.02)
            episode += 1
    vector_env.close()


if __name__ == "__main__":
    main()
