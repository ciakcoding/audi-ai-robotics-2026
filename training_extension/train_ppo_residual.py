from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)

from .td3_residual_env import RL_ACTION_NAMES, TD3BasketballResidualEnv


HERE = Path(__file__).resolve().parent


def make_env(rank, seed):
    def init():
        env = TD3BasketballResidualEnv()
        env.reset(seed=seed + rank)
        return env

    return init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name")
    args = parser.parse_args()

    run_name = args.run_name or datetime.now().strftime(
        "ppo_cem_residual_%Y%m%d_%H%M%S"
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
            "hoop_xy_error",
            "has_fallen",
            "smoothness_penalty",
        ),
    )
    vector_env = VecNormalize(
        vector_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    eval_env = VecMonitor(
        DummyVecEnv([make_env(10_000, args.seed)]),
        info_keywords=(
            "success",
            "hoop_xy_error",
            "has_fallen",
            "smoothness_penalty",
        ),
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        training=False,
        clip_obs=10.0,
    )

    model = PPO(
        "MlpPolicy",
        vector_env,
        learning_rate=2e-5,
        n_steps=512,
        batch_size=512,
        n_epochs=5,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.10,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.01,
        policy_kwargs={
            "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            "log_std_init": -3.0,
        },
        seed=args.seed,
        device=args.device,
        tensorboard_log=str(run_dir / "tensorboard"),
        verbose=1,
    )
    # Exact zero mean means deterministic policy == frozen CEM at step zero.
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
    model.save(run_dir / "initial_model")
    vector_env.save(run_dir / "vecnormalize_initial.pkl")

    metadata = {
        "algorithm": "PPO residual around frozen CEM v17",
        "timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "device_requested": args.device,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "action_names": RL_ACTION_NAMES,
        "target": [2.2, 0.0, 1.2],
        "success_radius": 0.10,
        "zero_mean_initialization": True,
        "log_std_init": -3.0,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    checkpoint = CheckpointCallback(
        save_freq=max(25_000 // args.n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_residual",
        save_vecnormalize=True,
    )
    evaluator = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "evaluations"),
        eval_freq=max(25_000 // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList([checkpoint, evaluator]),
        )
        model.save(run_dir / "final_model")
        vector_env.save(run_dir / "vecnormalize_final.pkl")
    finally:
        vector_env.close()
        eval_env.close()
    print(f"Saved residual PPO run: {run_dir}")


if __name__ == "__main__":
    main()
