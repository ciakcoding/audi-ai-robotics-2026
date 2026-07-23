"""Export the teacher-required PPO evaluation curve."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.run_dir / "evaluations" / "evaluations.npz")
    timesteps = data["timesteps"]
    results = data["results"]
    means = results.mean(axis=1)
    stds = results.std(axis=1)
    best_index = int(np.argmax(means))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(timesteps, means, color="#2c7bb6", label="Mean eval reward")
    ax.fill_between(
        timesteps,
        means - stds,
        means + stds,
        color="#2c7bb6",
        alpha=0.2,
        label="±1 std",
    )
    ax.scatter(
        [timesteps[best_index]],
        [means[best_index]],
        color="#d7191c",
        zorder=3,
        label=f"Best ({timesteps[best_index]:,} steps)",
    )
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Deterministic evaluation reward")
    ax.set_title("PPO evaluation curve - 100 episodes every 25k steps")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.run_dir / "training_evaluation_curve.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
