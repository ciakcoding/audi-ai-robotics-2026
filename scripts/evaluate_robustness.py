#!/usr/bin/env python3
"""
evaluate_robustness.py — Sim2Real Robustness Evaluation
=======================================================
Compares trained policy performance in clean vs. robustness (noisy) conditions.

Usage:
    # Default: compare clean vs all-noise
    .venv/Scripts/python scripts/evaluate_robustness.py

    # Custom policy and episodes
    .venv/Scripts/python scripts/evaluate_robustness.py \
        --policy policies/my_policy.zip \
        --episodes 100

    # Test only specific perturbations
    .venv/Scripts/python scripts/evaluate_robustness.py \
        --obs-noise 0.03 \
        --ball-mass 0.06 0.12 \
        --joint-friction 0.7 1.3

    # Save results to CSV for plotting
    .venv/Scripts/python scripts/evaluate_robustness.py --output results.csv

Output:
    Prints a comparison table (Clean vs Noisy) for:
      - Mean/Median/Min best distance to target
      - Hit rate < 30cm / < 50cm
      - Release-to-land time, landing position variance
"""

from __future__ import annotations
import argparse
import sys
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from envs.g1_robustness_env import G1RobustnessEnv

# Default to teammate's best model
DEFAULT_POLICY = str(ROOT / "outputs" / "models" / "selected" / "best" / "best_model.zip")
DEFAULT_SCENE  = str(ROOT / "assets" / "scene_throw.xml")


def parse_range_arg(s: str | None, key: str) -> tuple[float, float] | None:
    """Parse 'min max' string into tuple, e.g. '0.06 0.12' → (0.06, 0.12)."""
    if s is None:
        return None
    parts = s.strip().split()
    if len(parts) != 2:
        raise ValueError(f"--{key} requires two numbers, got: {s}")
    return float(parts[0]), float(parts[1])


def run_episodes(env, model, n_episodes: int, label: str, verbose: bool = True) -> dict:
    """Run N evaluation episodes and return statistics."""
    best_dists = []
    final_dists = []
    release_times = []
    landing_positions = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        best_dists.append(info.get("best_dist", np.inf))
        final_dists.append(info.get("dist_to_target", np.inf))
        release_times.append(info.get("release_time", 0))
        landing_positions.append(env._ball_pos().copy())

        if verbose and (ep + 1) % 20 == 0:
            print(f"  [{label}] {ep + 1}/{n_episodes} done | "
                  f"running best: {np.mean(best_dists):.3f}m")

    best = np.array(best_dists)
    final = np.array(final_dists)
    rt = np.array(release_times)
    landings = np.array(landing_positions)

    # Landing position spread (std dev of x, y)
    landing_std = np.std(landings[:, :2], axis=0)

    return {
        "label": label,
        "episodes": n_episodes,
        "mean_best_dist": float(np.mean(best)),
        "median_best_dist": float(np.median(best)),
        "min_best_dist": float(np.min(best)),
        "max_best_dist": float(np.max(best)),
        "std_best_dist": float(np.std(best)),
        "mean_final_dist": float(np.mean(final)),
        "hit_rate_30cm": float(100 * np.mean(best < 0.30)),
        "hit_rate_50cm": float(100 * np.mean(best < 0.50)),
        "mean_release_time": float(np.mean(rt)),
        "landing_x_std": float(landing_std[0]),
        "landing_y_std": float(landing_std[1]),
    }


def print_comparison(clean: dict, noisy: dict):
    """Print a formatted comparison table."""
    print()
    print("=" * 72)
    print("  SIM2REAL ROBUSTNESS EVALUATION")
    print("=" * 72)
    print()

    metrics = [
        ("Mean Best Distance (m)",  "mean_best_dist",    "{:.3f}",  "lower"),
        ("Median Best Distance (m)","median_best_dist",  "{:.3f}",  "lower"),
        ("Min Best Distance (m)",   "min_best_dist",     "{:.3f}",  "lower"),
        ("Max Best Distance (m)",   "max_best_dist",     "{:.3f}",  "lower"),
        ("Std Best Distance (m)",   "std_best_dist",     "{:.3f}",  "lower"),
        ("Mean Final Distance (m)", "mean_final_dist",   "{:.3f}",  "lower"),
        ("Hit Rate < 30cm (%)",    "hit_rate_30cm",     "{:.1f}",   "higher"),
        ("Hit Rate < 50cm (%)",    "hit_rate_50cm",     "{:.1f}",   "higher"),
        ("Landing X-Std (m)",      "landing_x_std",     "{:.4f}",   "lower"),
        ("Landing Y-Std (m)",      "landing_y_std",     "{:.4f}",   "lower"),
    ]

    print(f"  {'Metric':<28} {'Clean':>10} {'Noisy':>10} {'Delta':>10} {'Status'}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for name, key, fmt, direction in metrics:
        c = clean[key]
        n = noisy[key]
        delta = n - c
        if direction == "lower":
            arrow = "▲ WORSE" if delta > 0 else "▼ BETTER" if delta < 0 else "— SAME"
        else:
            arrow = "▼ WORSE" if delta < 0 else "▲ BETTER" if delta > 0 else "— SAME"
        print(f"  {name:<28} {fmt.format(c):>10} {fmt.format(n):>10} "
              f"{fmt.format(delta):>10} {arrow}")

    print()
    print(f"  Episodes per condition: {clean['episodes']}")
    print("=" * 72)
    print()

    # Summary for report
    if clean["hit_rate_30cm"] > 0 or noisy["hit_rate_30cm"] > 0:
        loss = clean["hit_rate_30cm"] - noisy["hit_rate_30cm"]
        print(f"  Performance loss (hit rate < 30cm): {loss:.1f}%")
    dist_loss = noisy["mean_best_dist"] - clean["mean_best_dist"]
    print(f"  Mean distance degradation: {dist_loss:.3f}m")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Sim2Real Robustness Evaluation (Clean vs Noisy)"
    )
    parser.add_argument(
        "--policy", type=str,
        default=DEFAULT_POLICY,
        help="Path to trained policy .zip"
    )
    parser.add_argument(
        "--scene", type=str,
        default=DEFAULT_SCENE,
        help="Path to scene XML"
    )
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes per condition")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON or CSV file")

    # Perturbation flags (each can be enabled independently)
    parser.add_argument("--obs-noise", type=float, default=0.02,
                        help="Observation noise std (0 = off)")
    parser.add_argument("--ball-mass", type=str, default="0.06 0.12",
                        help="Ball mass range: 'min max' (\"\" = off)")
    parser.add_argument("--joint-friction", type=str, default="0.7 1.3",
                        help="Joint friction multiplier range (\"\" = off)")
    parser.add_argument("--joint-damping", type=str, default=None,
                        help="Joint damping multiplier range (\"\" = off)")
    parser.add_argument("--floor-friction", type=str, default="0.5 1.5",
                        help="Floor friction multiplier range (\"\" = off)")
    parser.add_argument("--actuator-gain", type=str, default="0.85 1.0",
                        help="Actuator force range multiplier (\"\" = off)")
    parser.add_argument("--target-noise", type=float, default=0.05,
                        help="Target position noise std (m) (0 = off)")
    parser.add_argument("--latency-steps", type=int, default=3,
                        help="Control latency in sim steps (0 = off)")
    parser.add_argument("--enable-all", action="store_true",
                        help="Enable all perturbations with defaults")

    args = parser.parse_args()

    # ── Parse range arguments ──
    ball_mass = parse_range_arg(args.ball_mass, "ball-mass") if args.ball_mass else None
    joint_friction = parse_range_arg(args.joint_friction, "joint-friction") if args.joint_friction else None
    joint_damping = parse_range_arg(args.joint_damping, "joint-damping") if args.joint_damping else None
    floor_friction = parse_range_arg(args.floor_friction, "floor-friction") if args.floor_friction else None
    actuator_gain = parse_range_arg(args.actuator_gain, "actuator-gain") if args.actuator_gain else None

    # ── Check policy exists ──
    if not Path(args.policy).exists():
        print(f"ERROR: Policy not found: {args.policy}")
        print("Train first or specify with --policy")
        sys.exit(1)

    # ── Load model ──
    print(f"Loading policy: {args.policy}")
    model = PPO.load(args.policy)

    # ═══════════════════════════════════════════════════════════
    #  CLEAN evaluation (no randomization)
    # ═══════════════════════════════════════════════════════════
    print(f"\nRunning CLEAN evaluation ({args.episodes} episodes)...")
    env_clean = G1RobustnessEnv(enable_all=False)
    clean_results = run_episodes(env_clean, model, args.episodes, "CLEAN")

    # ═══════════════════════════════════════════════════════════
    #  NOISY evaluation (with randomization)
    # ═══════════════════════════════════════════════════════════
    print(f"\nRunning NOISY evaluation ({args.episodes} episodes)...")
    env_noisy = G1RobustnessEnv(
        enable_all=args.enable_all,
        obs_noise=args.obs_noise,
        ball_mass_range=ball_mass,
        joint_friction_range=joint_friction,
        joint_damping_range=joint_damping,
        floor_friction_range=floor_friction,
        actuator_gain_range=actuator_gain,
        target_pos_noise=args.target_noise,
        control_latency_steps=args.latency_steps,
    )

    # Print what's being randomized
    noisy_params = []
    if args.obs_noise > 0:
        noisy_params.append(f"obs_noise={args.obs_noise}")
    if ball_mass:
        noisy_params.append(f"ball_mass={ball_mass}")
    if joint_friction:
        noisy_params.append(f"joint_friction={joint_friction}")
    if joint_damping:
        noisy_params.append(f"joint_damping={joint_damping}")
    if floor_friction:
        noisy_params.append(f"floor_friction={floor_friction}")
    if actuator_gain:
        noisy_params.append(f"actuator_gain={actuator_gain}")
    if args.target_noise > 0:
        noisy_params.append(f"target_noise={args.target_noise}")
    if args.latency_steps > 0:
        noisy_params.append(f"latency={args.latency_steps} steps")
    print(f"  Active: {', '.join(noisy_params) if noisy_params else 'none'}")

    noisy_results = run_episodes(env_noisy, model, args.episodes, "NOISY")

    # ═══════════════════════════════════════════════════════════
    #  COMPARE
    # ═══════════════════════════════════════════════════════════
    print_comparison(clean_results, noisy_results)

    # ── Save results ──
    if args.output:
        output_path = Path(args.output)
        if output_path.suffix == ".json":
            with open(output_path, "w") as f:
                json.dump({"clean": clean_results, "noisy": noisy_results}, f, indent=2)
        elif output_path.suffix == ".csv":
            import csv
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["condition", "metric", "value"])
                for key, val in clean_results.items():
                    if key != "label" and key != "episodes":
                        writer.writerow(["clean", key, val])
                for key, val in noisy_results.items():
                    if key != "label" and key != "episodes":
                        writer.writerow(["noisy", key, val])
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
