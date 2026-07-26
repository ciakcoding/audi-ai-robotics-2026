from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.noise import NormalActionNoise
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
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name")
    args = parser.parse_args()

    run_name = args.run_name or datetime.now().strftime(
        "td3_cem_residual_%Y%m%d_%H%M%S"
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

    action_noise = NormalActionNoise(
        mean=np.zeros(len(RL_ACTION_NAMES), dtype=np.float32),
        sigma=0.05 * np.ones(len(RL_ACTION_NAMES), dtype=np.float32),
    )
    model = TD3(
        "MlpPolicy",
        vector_env,
        learning_rate=5e-5,
        buffer_size=500_000,
        learning_starts=5_000,
        batch_size=512,
        tau=0.005,
        gamma=0.995,
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=action_noise,
        policy_delay=4,
        target_policy_noise=0.10,
        target_noise_clip=0.25,
        policy_kwargs={"net_arch": [256, 256]},
        seed=args.seed,
        device=args.device,
        tensorboard_log=str(run_dir / "tensorboard"),
        verbose=1,
    )
    # The residual policy must start exactly at the frozen CEM expert.
    # SB3 otherwise initializes a non-zero deterministic actor.
    with torch.no_grad():
        output_layer = model.policy.actor.mu[-2]
        output_layer.weight.zero_()
        output_layer.bias.zero_()
    model.policy.actor_target.load_state_dict(model.policy.actor.state_dict())
    model.save(run_dir / "initial_model")
    vector_env.save(run_dir / "vecnormalize_initial.pkl")

    metadata = {
        "algorithm": "TD3",
        "timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "device_requested": args.device,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "expert": "frozen/cem_v17_no_cross_20260725",
        "action_names": RL_ACTION_NAMES,
        "target": [2.2, 0.0, 1.2],
        "success_radius": 0.10,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    checkpoint = CheckpointCallback(
        save_freq=max(25_000 // args.n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="td3_residual",
        save_replay_buffer=True,
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
    print(f"Saved TD3 run: {run_dir}")


if __name__ == "__main__":
    main()
