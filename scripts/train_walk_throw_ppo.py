#!/usr/bin/env python3
"""Train Level 3 residual PPO: walk forward + throw ball."""

import argparse, json, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from envs.walk_throw_ppo_env import WalkThrowPPOEnv


def make_env(rank: int, seed: int):
    def _init():
        env = WalkThrowPPOEnv()
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-envs", type=int, default=4)
    args = parser.parse_args()

    seed = 2026
    run_name = datetime.now().strftime("walk_throw_%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs" / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [make_env(i, seed) for i in range(args.n_envs)]
    env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)
    info_keywords = ("success", "landing_error_xy", "has_fallen", "total_forward_distance_m")
    env = VecMonitor(env, str(run_dir / "monitor.csv"), info_keywords=info_keywords)

    eval_env = VecMonitor(
        DummyVecEnv([make_env(10000, seed)]),
        info_keywords=info_keywords,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=1.0,
        policy_kwargs={"net_arch": [512, 512, 256]},
        seed=seed,
        tensorboard_log=str(run_dir / "tensorboard"),
        device=args.device,
        verbose=1,
    )

    checkpoint = CheckpointCallback(
        save_freq=max(50000 // args.n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="walk_throw",
    )

    evaluator = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "evaluations"),
        eval_freq=max(50000 // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    metadata = {
        "level": 3,
        "method": "residual PPO (baseline = stand + throw, PPO learns walk)",
        "timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "device": args.device,
        "controlled_joints": 22,
        "residual_scale": 1.0,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Level 3 Residual PPO: walk + throw")
    print(f"  Timesteps: {args.timesteps}, Envs: {args.n_envs}, Device: {args.device}")
    print(f"  Baseline: stand + scripted arm throw")
    print(f"  PPO: learns residual to walk forward while throwing")
    print(f"  Run: {run_dir}")
    print()

    try:
        model.learn(total_timesteps=args.timesteps, callback=[checkpoint, evaluator])
    except KeyboardInterrupt:
        print("\nInterrupted. Saving...")

    model.save(str(run_dir / "final_model"))
    print(f"Saved to {run_dir}/final_model.zip")


if __name__ == "__main__":
    main()
