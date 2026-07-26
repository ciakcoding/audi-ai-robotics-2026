from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from .basketball_env import BasketballResidualEnv


HERE = Path(__file__).resolve().parent
CONTROL_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_elbow_joint",
    "right_wrist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_elbow_joint",
    "left_wrist_pitch_joint",
    "waist_pitch_joint",
]
PARAMETER_NAMES = (
    [f"{name}:load" for name in CONTROL_JOINT_NAMES]
    + [f"{name}:release" for name in CONTROL_JOINT_NAMES]
    + ["release_timing"]
)

# Residual action limits.  The peer trajectory remains the centre; these hard
# limits prevent the optimiser from finding visually invalid shoulder/wrist
# twists while leaving pitch and elbow motion expressive enough to throw.
ACTION_LIMITS = {
    "waist_pitch_joint": 0.30,
}


def controller_action(env: BasketballResidualEnv, parameters: np.ndarray):
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    index = {name: i for i, name in enumerate(env.control_joint_names)}
    step = env.policy.step_count
    count = len(CONTROL_JOINT_NAMES)
    load = parameters[:count]
    release = parameters[count : 2 * count]
    if step <= 350:
        offsets = np.zeros(count)
    elif step <= 380:
        u = (step - 350) / 30.0
        smooth = 3.0 * u * u - 2.0 * u * u * u
        offsets = smooth * load
    elif step <= 410:
        u = (step - 380) / 30.0
        smooth = 3.0 * u * u - 2.0 * u * u * u
        offsets = (1.0 - smooth) * load + smooth * release
    elif step <= 416:
        u = (step - 410) / 6.0
        smooth = 3.0 * u * u - 2.0 * u * u * u
        offsets = (1.0 - smooth) * release
    else:
        offsets = np.zeros(count)
    if step > 410:
        # The ball has already left the right hand.  Drop the learned guide-arm
        # residual immediately so the baseline can peel that arm outward
        # without first crossing the shooting forearm.
        for name in [
            "left_shoulder_pitch_joint",
            "left_elbow_joint",
            "left_wrist_pitch_joint",
        ]:
            offsets[CONTROL_JOINT_NAMES.index(name)] = 0.0
    for name, value in zip(CONTROL_JOINT_NAMES, offsets):
        limit = ACTION_LIMITS.get(name, 1.00)
        action[index[name]] = np.clip(value, -limit, limit)
    action[-1] = parameters[-1]
    return action


def evaluate(parameters, seeds):
    # Keep the teammate baseline's complete walk -> dip -> extension -> throw.
    env = BasketballResidualEnv(curriculum_radius=0.10, set_shot_only=False)
    scores = []
    records = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action = controller_action(env, parameters)
            obs, _, terminated, truncated, info = env.step(action)
        error = float(
            info["crossing_xy_error"]
            if info["crossing_xy_error"] is not None
            else info["hoop_xy_error"]
        )
        score = error
        if not info["crossed_hoop_plane"]:
            score += 0.75
        if info["touched_backboard"]:
            score += 1.0
        if info["has_fallen"]:
            score += 2.0
        release_distance = info["release_distance_to_hoop_xy"]
        if release_distance is None or release_distance < 1.10:
            score += 2.0
        pelvis_distance = info["release_pelvis_distance_to_hoop_xy"]
        if pelvis_distance is None or pelvis_distance < 1.20:
            score += 3.0
        if info["airborne_horizontal_distance"] < 1.00:
            score += 2.0
        if info["minimum_hand_to_hoop_distance"] < 0.45:
            score += 2.0
        if (
            info["release_ball_position"] is None
            or info["release_ball_position"][2] < 1.20
        ):
            score += 3.0
        if (
            info["release_hand_separation"] is None
            or info["release_hand_separation"] > 0.25
        ):
            score += 2.0
        if info["release_hand_separation"] is not None:
            score += 2.0 * max(
                0.0, info["release_hand_separation"] - 0.18
            )
        release_pos = info["release_ball_position"]
        release_vel = info["release_ball_velocity"]
        if release_pos is not None and release_vel is not None:
            pos = np.asarray(release_pos, dtype=np.float64)
            vel = np.asarray(release_vel, dtype=np.float64)
            height = float(pos[2] - env.target[2])
            gravity = abs(float(env.model.opt.gravity[2]))
            discriminant = vel[2] ** 2 + 2.0 * gravity * height
            if discriminant > 0.0:
                flight_time = (vel[2] + np.sqrt(discriminant)) / gravity
                predicted_xy = pos[:2] + vel[:2] * flight_time
                score += float(np.linalg.norm(predicted_xy - env.target[:2]))
            else:
                score += 2.0
            score += max(0.0, 0.5 - float(vel[0]))
            score += 0.25 * abs(float(vel[1]))
        score += 0.005 * float(np.dot(parameters[:-1], parameters[:-1]))
        scores.append(score)
        records.append(info)
    env.close()
    # Robust objective: improving only the average produced policies that
    # missed under a minority of reset perturbations.  Optimisation now pays
    # explicit attention to the worst training seed as well.
    return float(np.mean(scores) + 0.5 * np.max(scores)), records


def evaluate_payload(payload):
    parameters, seeds = payload
    return evaluate(parameters, seeds)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sigma-scale", type=float, default=0.5)
    parser.add_argument("--twohand-focus", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=False)
    source_snapshot = args.output / "source_snapshot"
    source_snapshot.mkdir()
    for source_name in [
        "optimize_direct.py",
        "basketball_env.py",
        "derived_baseline.py",
        "scene_throw_LEVEL03_ring.xml",
        "validate_direct.py",
        "render_direct.py",
        "config.json",
    ]:
        source = HERE / source_name
        if source.exists():
            (source_snapshot / source_name).write_bytes(source.read_bytes())
    iteration_milestones = args.output / "iteration_milestones"
    iteration_milestones.mkdir()
    rng = np.random.default_rng(args.seed)
    mean = np.zeros(len(PARAMETER_NAMES))
    sigma = np.array([0.45] * (len(PARAMETER_NAMES) - 1) + [0.40])
    if args.resume:
        state = json.loads(args.resume.read_text(encoding="utf-8"))
        old_parameters = dict(
            zip(state["parameter_names"], state["best_parameters"])
        )
        mean = np.asarray(
            [old_parameters.get(name, 0.0) for name in PARAMETER_NAMES],
            dtype=np.float64,
        )
        sigma *= args.resume_sigma_scale
    if args.twohand_focus:
        sigma[:] = 0.008
        count = len(CONTROL_JOINT_NAMES)
        left_indices = [
            CONTROL_JOINT_NAMES.index(name)
            for name in [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ]
        ]
        for index in left_indices:
            sigma[index] = 0.05
            sigma[count + index] = 0.16
    seeds = list(range(args.seed, args.seed + args.seed_count))
    best_score = np.inf
    best_parameters = mean.copy()
    history = []

    with mp.get_context("spawn").Pool(args.workers) as pool:
        for iteration in range(args.iterations):
            samples = np.clip(
                mean + rng.normal(size=(args.population, len(mean))) * sigma,
                -1.0,
                1.0,
            )
            # Always retain the current center as an explicit candidate.
            samples[0] = mean
            results = pool.map(
                evaluate_payload, [(sample, seeds) for sample in samples]
            )
            order = np.argsort(results)
            elite_count = max(4, args.population // 5)
            elites = samples[order[:elite_count]]
            weights = np.linspace(elite_count, 1, elite_count, dtype=np.float64)
            weights /= weights.sum()
            mean = np.sum(elites * weights[:, None], axis=0)
            elite_std = np.sqrt(
                np.sum(weights[:, None] * (elites - mean) ** 2, axis=0)
            )
            sigma = np.maximum(0.7 * sigma + 0.3 * elite_std, 0.003)
            if results[order[0]] < best_score:
                best_score = float(results[order[0]])
                best_parameters = samples[order[0]].copy()
            entry = {
                "iteration": iteration + 1,
                "best_score": best_score,
                "generation_score": float(results[order[0]]),
                "best_parameters": best_parameters.tolist(),
                "mean": mean.tolist(),
                "sigma": sigma.tolist(),
            }
            history.append(entry)
            (args.output / "state.json").write_text(
                json.dumps(
                    {
                        **entry,
                        "parameter_names": PARAMETER_NAMES,
                        "seeds": seeds,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (args.output / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            (iteration_milestones / f"iteration_{iteration + 1:04d}.json").write_text(
                json.dumps(
                    {
                        **entry,
                        "parameter_names": PARAMETER_NAMES,
                        "training_seeds": seeds,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"iteration={iteration + 1} best={best_score:.5f} "
                f"generation={results[order[0]]:.5f}",
                flush=True,
            )

    score, records = evaluate(best_parameters, range(3000, 3020))
    summary = {
        "validation_score": score,
        "successes": sum(int(record["success"]) for record in records),
        "episodes": len(records),
        "mean_error": float(
            np.mean([record["hoop_xy_error"] for record in records])
        ),
        "backboard_contacts": sum(
            int(record["touched_backboard"]) for record in records
        ),
        "falls": sum(int(record["has_fallen"]) for record in records),
        "best_parameters": best_parameters.tolist(),
        "parameter_names": PARAMETER_NAMES,
    }
    (args.output / "validation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
