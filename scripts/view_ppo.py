"""Continuously display the selected Task 2 PPO policy."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco.viewer
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.ppo_throw_env import PPOThrowEnv


DEFAULT_MODEL = (
    ROOT
    / "outputs"
    / "models"
    / "selected"
    / "best"
    / "best_model.zip"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    env = PPOThrowEnv(residual_scale=0.2)
    model = PPO.load(args.model, device="cpu")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        episode = 0
        while viewer.is_running():
            obs, _ = env.reset(seed=args.seed + episode)
            terminated = truncated = False
            info = {}
            while not (terminated or truncated) and viewer.is_running():
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                viewer.sync()
                time.sleep(env.control_dt)
            error = info.get("landing_error_xy")
            if error is not None:
                print(
                    f"episode={episode} landing_error={error * 100:.3f} cm "
                    f"success={info['success']} fallen={info['has_fallen']}"
                )
            episode += 1
    env.close()


if __name__ == "__main__":
    main()
