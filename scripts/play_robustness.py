#!/usr/bin/env python3
"""
play_robustness.py — Visual Sim2Real Robustness Playback
========================================================
Watch the trained policy with visible Sim2Real perturbations:

    Random pushes to the ball    → trajectory visibly changes
    Actuator weakness            → arm moves slower / softer
    Target position randomized   → different target each episode

Usage:
    .venv/Scripts/python scripts/play_robustness.py
    .venv/Scripts/python scripts/play_robustness.py --push-force 10 --push-prob 0.01
    .venv/Scripts/python scripts/play_robustness.py --enable-all

Controls (in viewer window):
    drag        = rotate camera
    ctrl+drag   = pan
    scroll      = zoom
    space       = pause / resume
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from envs.g1_robustness_env import G1RobustnessEnv

DEFAULT_POLICY = str(ROOT / "outputs" / "models" / "selected" / "best" / "best_model.zip")
DEFAULT_SCENE  = str(ROOT / "assets" / "scene_throw.xml")


def parse_range_arg(s: str | None, key: str) -> tuple[float, float] | None:
    if s is None:
        return None
    parts = s.strip().split()
    if len(parts) != 2:
        raise ValueError(f"--{key} requires two numbers, got: {s}")
    return float(parts[0]), float(parts[1])


def main():
    parser = argparse.ArgumentParser(
        description="Visual Sim2Real Robustness Playback"
    )
    parser.add_argument("--policy", type=str, default=DEFAULT_POLICY)
    parser.add_argument("--scene", type=str, default=DEFAULT_SCENE)
    parser.add_argument("--enable-all", action="store_true",
                        help="Enable all perturbations")
    parser.add_argument("--obs-noise", type=float, default=0.0,
                        help="Observation noise std")
    parser.add_argument("--ball-mass", type=str, default=None,
                        help="Ball mass range 'min max'")
    parser.add_argument("--joint-friction", type=str, default=None,
                        help="Joint friction range 'min max'")
    parser.add_argument("--floor-friction", type=str, default=None)
    parser.add_argument("--actuator-gain", type=str, default="0.85 1.0")
    parser.add_argument("--target-noise", type=float, default=0.0)
    parser.add_argument("--push-force", type=float, default=5.0,
                        help="Max push force (N) on ball")
    parser.add_argument("--push-prob", type=float, default=0.005,
                        help="Push probability per step")
    parser.add_argument("--latency-steps", type=int, default=0)
    parser.add_argument("--show-randomization", action="store_true",
                        help="Print randomization values each episode")

    args = parser.parse_args()

    if not Path(args.policy).exists():
        print(f"Policy not found: {args.policy}")
        sys.exit(1)

    ball_mass = parse_range_arg(args.ball_mass, "ball-mass") if args.ball_mass else None
    joint_friction = parse_range_arg(args.joint_friction, "joint-friction") if args.joint_friction else None
    floor_friction = parse_range_arg(args.floor_friction, "floor-friction") if args.floor_friction else None
    actuator_gain = parse_range_arg(args.actuator_gain, "actuator-gain") if args.actuator_gain else None

    env = G1RobustnessEnv(
        enable_all=args.enable_all,
        obs_noise=args.obs_noise,
        ball_mass_range=ball_mass,
        joint_friction_range=joint_friction,
        floor_friction_range=floor_friction,
        actuator_gain_range=actuator_gain,
        target_pos_noise=args.target_noise,
        push_probability=args.push_prob,
        push_force_range=(-args.push_force, args.push_force),
        control_latency_steps=args.latency_steps,
    )

    model = PPO.load(args.policy)
    print(f"Policy loaded: {args.policy}")
    print(f"Push force: ±{args.push_force}N | Push prob: {args.push_prob}/step")
    if args.enable_all:
        print("ALL perturbations enabled")
    print("Close viewer window to exit.")

    obs, info = env.reset()
    episode = 0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.azimuth = 140
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0

        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            viewer.sync()
            time.sleep(env.control_dt)

            if terminated or truncated:
                episode += 1
                print(f"Ep {episode} | best_dist={info.get('best_dist', np.inf):.3f}m "
                      f"| released={info.get('released', False)}")
                if args.show_randomization:
                    env.print_randomization()

                obs, info = env.reset()


if __name__ == "__main__":
    main()
