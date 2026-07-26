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
                start = time.perf_counter()
                action = controller_action(env, parameters)
                _, _, terminated, truncated, info = env.step(action)
                viewer.sync()
                remaining = 0.02 - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)
            if not viewer.is_running():
                break
            print(
                "Ball reached target: "
                f"{'YES' if info['success'] else 'NO'} | "
                f"Distance from hoop centre at crossing="
                f"{info['crossing_xy_error'] * 100:.2f} cm | "
                f"Airborne distance="
                f"{info['airborne_horizontal_distance']:.2f} m | "
                f"Backboard contact="
                f"{'YES' if info['touched_backboard'] else 'NO'} | "
                f"Robot fell={'YES' if info['has_fallen'] else 'NO'}"
            )
            end_hold = time.perf_counter() + args.hold_seconds
            while viewer.is_running() and time.perf_counter() < end_hold:
                viewer.sync()
                time.sleep(0.02)
            episode += 1
    vector_env.close()


if __name__ == "__main__":
    main()
