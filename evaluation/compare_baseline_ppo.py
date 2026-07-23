"""Fair frozen-world comparison of baseline, best PPO, and final PPO."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.ppo_throw_env import PPOThrowEnv


def run_policy(env, policy, episodes: int, seed: int):
    rows = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        terminated = truncated = False
        reward_sum = 0.0
        info = {}
        while not (terminated or truncated):
            if policy is None:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
                action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
        rows.append(
            {
                "episode": episode,
                "seed": seed + episode,
                "success": bool(info["success"]),
                "landing_error_xy_m": info["landing_error_xy"],
                "landing_x_m": None
                if info["landing_pos"] is None
                else float(info["landing_pos"][0]),
                "landing_y_m": None
                if info["landing_pos"] is None
                else float(info["landing_pos"][1]),
                "has_fallen": bool(info["has_fallen"]),
                "torso_tilt_deg": float(info["torso_tilt_deg"]),
                "reward_sum": float(reward_sum),
            }
        )
    return rows


def summarize(rows):
    errors = np.array(
        [
            row["landing_error_xy_m"]
            for row in rows
            if row["landing_error_xy_m"] is not None
        ],
        dtype=float,
    )
    return {
        "episodes": len(rows),
        "landed_episodes": int(len(errors)),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_landing_error_xy_m": float(np.mean(errors)) if len(errors) else None,
        "p90_landing_error_xy_m": float(np.percentile(errors, 90))
        if len(errors)
        else None,
        "max_landing_error_xy_m": float(np.max(errors)) if len(errors) else None,
        "fall_rate": float(np.mean([row["has_fallen"] for row in rows])),
        "mean_reward": float(np.mean([row["reward_sum"] for row in rows])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--joint-noise", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "metrics")
    args = parser.parse_args()

    suffix = (
        "nominal"
        if args.joint_noise == 0.0
        else f"joint_noise_{args.joint_noise:.3f}".replace(".", "p")
    )
    output = args.output_dir / f"comparison_{suffix}"
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(
        (args.run_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    residual_scale = float(metadata["config"]["residual_scale"])
    env = PPOThrowEnv(
        residual_scale=residual_scale,
        extra_initial_joint_noise=args.joint_noise,
    )
    policies = {
        "baseline": None,
        "ppo_best": PPO.load(args.run_dir / "best" / "best_model.zip", device="cpu"),
        "ppo_final": PPO.load(args.run_dir / "final_model.zip", device="cpu"),
    }
    all_rows = {}
    summary = {
        "protocol": {
            "seed_start": args.seed,
            "episodes": args.episodes,
            "task1_commit": "7a370663cbcc1aa96438dffc9f6331d3bf4ef35c",
            "target_xy_m": [0.55, 0.0],
            "success_radius_m": 0.1,
            "release_time_s": 0.65,
            "extra_initial_joint_noise_rad": args.joint_noise,
        }
    }
    for name, policy in policies.items():
        rows = run_policy(env, policy, args.episodes, args.seed)
        all_rows[name] = rows
        summary[name] = summarize(rows)
    env.close()

    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["policy", *list(next(iter(all_rows.values()))[0])]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, rows in all_rows.items():
            for row in rows:
                writer.writerow({"policy": name, **row})
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    names = ["baseline", "ppo_best", "ppo_final"]
    labels = ["Baseline", "PPO best", "PPO final"]
    errors_cm = [summary[name]["mean_landing_error_xy_m"] * 100 for name in names]
    success_pct = [summary[name]["success_rate"] * 100 for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, errors_cm, color=["#777777", "#d7191c", "#2c7bb6"])
    axes[0].set_ylabel("Mean landing error (cm)")
    axes[0].set_title("Lower is better")
    axes[1].bar(labels, success_pct, color=["#777777", "#d7191c", "#2c7bb6"])
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Success rate (%)")
    axes[1].set_title("Frozen 10 cm success radius")
    fig.suptitle("Baseline vs PPO - identical 100-seed protocol")
    fig.tight_layout()
    fig.savefig(output / "baseline_vs_ppo.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, label in zip(names, labels):
        values = np.sort(
            [
                row["landing_error_xy_m"] * 100
                for row in all_rows[name]
                if row["landing_error_xy_m"] is not None
            ]
        )
        ax.plot(values, np.arange(1, len(values) + 1) / len(values), label=label)
    ax.set_xlabel("Landing error (cm)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Landing-error distribution")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "landing_error_cdf.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
