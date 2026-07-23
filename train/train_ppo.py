"""Train PPO entirely inside the isolated Task 2 directory."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.ppo_throw_env import PPOThrowEnv


def make_env(
    rank: int,
    seed: int,
    residual_scale: float,
    extra_initial_joint_noise: float,
):
    def init():
        env = PPOThrowEnv(
            residual_scale=residual_scale,
            extra_initial_joint_noise=extra_initial_joint_noise,
        )
        env.reset(seed=seed + rank)
        return env

    return init


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    config = json.loads((ROOT / "configs" / "ppo.json").read_text(encoding="utf-8"))
    run_name = args.run_name or datetime.now().strftime("ppo_%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs" / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "task1_commit": "7a370663cbcc1aa96438dffc9f6331d3bf4ef35c",
                "task1_tag": "task1-baseline-v1.0",
                "timesteps": args.timesteps,
                "config": config,
                "python": sys.version,
                "platform": platform.platform(),
                "training_status": "running",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    n_envs = int(config["n_envs"])
    env_fns = [
        make_env(
            rank,
            config["seed"],
            config["residual_scale"],
            config["extra_initial_joint_noise"],
        )
        for rank in range(n_envs)
    ]
    env = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)
    info_keywords = (
        "success",
        "landing_error_xy",
        "has_fallen",
        "task2_base_reward",
        "task2_residual_l2",
    )
    env = VecMonitor(
        env,
        str(run_dir / "monitor.csv"),
        info_keywords=info_keywords,
    )
    eval_env = VecMonitor(
        DummyVecEnv(
            [
                make_env(
                    10_000,
                    config["seed"],
                    config["residual_scale"],
                    config["extra_initial_joint_noise"],
                )
            ]
        ),
        info_keywords=info_keywords,
    )
    model = PPO(
        config["policy"],
        env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        policy_kwargs={"net_arch": config["policy_net"]},
        seed=config["seed"],
        tensorboard_log=str(run_dir / "tensorboard"),
        device=config["device"],
        verbose=1,
    )
    checkpoint = CheckpointCallback(
        save_freq=max(int(config["checkpoint_freq"]) // n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_throw",
    )
    evaluator = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "evaluations"),
        eval_freq=max(int(config["eval_freq"]) // n_envs, 1),
        n_eval_episodes=int(config["eval_episodes"]),
        deterministic=True,
        render=False,
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList([checkpoint, evaluator]),
    )
    model.save(run_dir / "final_model")
    env.close()
    eval_env.close()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["training_status"] = "completed"
    metadata["wall_time_seconds"] = time.time() - started
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    latest = ROOT / "runs" / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_dir():
            shutil.rmtree(latest)
        else:
            latest.unlink()
    shutil.copytree(run_dir, latest)
    print(f"Saved PPO run: {run_dir}")


if __name__ == "__main__":
    main()
