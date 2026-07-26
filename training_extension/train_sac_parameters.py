from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)

from .optimize_direct import PARAMETER_NAMES
from .sac_parameter_env import SACShotParameterEnv


HERE = Path(__file__).resolve().parent


def make_env(rank, seed):
    def init():
        env = SACShotParameterEnv()
        env.reset(seed=seed + rank)
        return env

    return init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2040)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name")
    args = parser.parse_args()

    run_name = args.run_name or datetime.now().strftime(
        "sac_cem_parameters_%Y%m%d_%H%M%S"
    )
    run_dir = HERE / "runs" / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    env_fns = [make_env(i, args.seed) for i in range(args.n_envs)]
    vector_env = (
        SubprocVecEnv(env_fns)
        if args.n_envs > 1
        else DummyVecEnv(env_fns)
    )
    vector_env = VecMonitor(
        vector_env,
        str(run_dir / "monitor.csv"),
        info_keywords=(
            "success",
            "crossing_xy_error",
            "touched_backboard",
            "has_fallen",
            "parameter_l2",
            "mean_ctrl_delta",
        ),
    )
    vector_env = VecNormalize(
        vector_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    model = SAC(
        "MlpPolicy",
        vector_env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=512,
        batch_size=256,
        tau=0.005,
        gamma=0.0,
        train_freq=(1, "step"),
        gradient_steps=args.n_envs,
        ent_coef=0.005,
        policy_kwargs={
            "net_arch": {"pi": [256, 256], "qf": [256, 256]},
            "log_std_init": -2.3,
        },
        seed=args.seed,
        device=args.device,
        tensorboard_log=str(run_dir / "tensorboard"),
        verbose=1,
    )
    # Deterministic step-zero policy exactly reproduces frozen CEM v17.
    with torch.no_grad():
        model.actor.mu.weight.zero_()
        model.actor.mu.bias.zero_()
    model.save(run_dir / "initial_model")
    vector_env.save(run_dir / "vecnormalize_initial.pkl")

    metadata = {
        "algorithm": "SAC parameter residual around frozen CEM v17",
        "timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "device_requested": args.device,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "parameter_names": PARAMETER_NAMES,
        "target": [2.2, 0.0, 1.2],
        "success_radius": 0.10,
        "parameter_scales": [
            0.005, 0.002, 0.005, 0.005, 0.002, 0.005, 0.010,
            0.010, 0.005, 0.005, 0.005, 0.002, 0.005, 0.010, 0.0,
        ],
        "zero_mean_initialization": True,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    checkpoint = CheckpointCallback(
        save_freq=max(1_000 // args.n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="sac_parameters",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    try:
        model.learn(total_timesteps=args.timesteps, callback=checkpoint)
        model.save(run_dir / "final_model")
        model.save_replay_buffer(run_dir / "replay_buffer.pkl")
        vector_env.save(run_dir / "vecnormalize_final.pkl")
    finally:
        vector_env.close()
    print(f"Saved parameter SAC run: {run_dir}")


if __name__ == "__main__":
    main()
