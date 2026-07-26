from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)

from .train_ppo_parameters import HERE, make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vecnormalize", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=4_096)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2052)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--clip-range", type=float, default=0.03)
    parser.add_argument("--target-kl", type=float, default=0.005)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--run-name")
    args = parser.parse_args()

    run_name = args.run_name or datetime.now().strftime(
        "ppo_cem_parameters_finetune_%Y%m%d_%H%M%S"
    )
    run_dir = HERE / "runs" / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    env_fns = [make_env(i, args.seed) for i in range(args.n_envs)]
    vector_env = VecMonitor(
        SubprocVecEnv(env_fns),
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
    vector_env = VecNormalize.load(args.vecnormalize, vector_env)
    vector_env.training = True
    vector_env.norm_reward = False

    model = PPO.load(args.model, env=vector_env, device=args.device)
    # Fine-tune conservatively around the already validated 1,024-shot policy.
    model.lr_schedule = lambda _: args.learning_rate
    for group in model.policy.optimizer.param_groups:
        group["lr"] = args.learning_rate
    model.clip_range = lambda _: args.clip_range
    model.target_kl = args.target_kl
    model.n_epochs = args.n_epochs
    model.tensorboard_log = str(run_dir / "tensorboard")
    model.verbose = 1
    model.save(run_dir / "initial_model")
    vector_env.save(run_dir / "vecnormalize_initial.pkl")

    metadata = {
        "algorithm": "PPO parameter residual fine-tune",
        "parent_model": str(args.model.resolve()),
        "parent_vecnormalize": str(args.vecnormalize.resolve()),
        "parent_timesteps": int(model.num_timesteps),
        "additional_timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "clip_range": args.clip_range,
        "target_kl": args.target_kl,
        "n_epochs": args.n_epochs,
        "device_requested": args.device,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "target": [2.2, 0.0, 1.2],
        "success_radius": 0.10,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    checkpoint = CheckpointCallback(
        save_freq=max(512 // args.n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_parameters_finetune",
        save_vecnormalize=True,
    )
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint,
            reset_num_timesteps=False,
        )
        model.save(run_dir / "final_model")
        vector_env.save(run_dir / "vecnormalize_final.pkl")
    finally:
        vector_env.close()
    print(f"Saved parameter PPO fine-tune: {run_dir}")


if __name__ == "__main__":
    main()
