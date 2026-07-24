"""Extended, reproducible evaluation for the frozen Task 1 world.

This script compares the scripted residual baseline (zero residual action) with
the selected PPO best checkpoint. It does not change the frozen task success
contract. Additional post-contact and stability values are diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.ppo_throw_env import PPOThrowEnv


TASK1_COMMIT = "7a370663cbcc1aa96438dffc9f6331d3bf4ef35c"
TARGET_XY_M = np.array([0.55, 0.0], dtype=np.float64)
SUCCESS_RADIUS_M = 0.10
RELEASE_TIME_CONFIG_S = 0.65
POST_CONTACT_DIAGNOSTIC_S = 3.0
SETTLED_SPEED_MPS = 0.02
SETTLED_WINDOW_S = 0.20
DOCUMENT_TILT_DIAGNOSTIC_DEG = 2.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ObservedPPOThrowEnv(PPOThrowEnv):
    """PPO environment with physics-substep instrumentation."""

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.observed_hold_max_tilt_deg = self._torso_tilt_deg()
        self.observed_release_tilt_deg = None
        self.observed_episode_max_tilt_deg = self.observed_hold_max_tilt_deg
        self.observed_post_release_tilts: list[tuple[float, float]] = []
        self.observed_impact_velocity_z_mps = None
        self.observed_peak_impact_force_n = 0.0
        self.observed_invalid_flight_contacts: set[str] = set()
        self.observed_first_contact_time_s = None
        return obs, info

    def _torso_tilt_deg(self) -> float:
        up = float(
            np.clip(
                self.data.xmat[self.torso_body_id].reshape(3, 3)[2, 2],
                -1.0,
                1.0,
            )
        )
        return float(np.degrees(np.arccos(up)))

    def _scan_contacts(self, pre_step_ball_vz: float) -> None:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair == {self.ball_geom_id, self.floor_geom_id}:
                if self.observed_impact_velocity_z_mps is None:
                    self.observed_impact_velocity_z_mps = float(pre_step_ball_vz)
                    self.observed_first_contact_time_s = float(self.data.time)
                force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, index, force)
                self.observed_peak_impact_force_n = max(
                    self.observed_peak_impact_force_n, float(max(force[0], 0.0))
                )
            elif self.released and self.ball_geom_id in pair:
                other = (
                    int(contact.geom2)
                    if int(contact.geom1) == self.ball_geom_id
                    else int(contact.geom1)
                )
                name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, other
                )
                self.observed_invalid_flight_contacts.add(name or f"geom_{other}")

    def step(self, action):
        was_released = self.released
        residual = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        progress = min(1.0, self.step_count / 34.0)
        baseline = self.baseline_start + progress * (
            self.baseline_end - self.baseline_start
        )
        applied = np.zeros(8, dtype=np.float64)
        applied[:7] = np.clip(
            baseline + self.residual_scale * residual, -1.0, 1.0
        )

        self.data.ctrl[:] = self.nominal_ctrl
        self.data.ctrl[self.arm_actuator_ids] = (
            self.nominal_ctrl[self.arm_actuator_ids]
            + self.action_scale * applied[: self.n_arm]
        )
        t = self.step_count * self.control_dt
        if not self.released and t >= self.scripted_release_time:
            self.data.eq_active[self.hold_eq_id] = 0
            self.released = True
            self.release_time = t
            self.observed_release_tilt_deg = self._torso_tilt_deg()

        for _ in range(self.frame_skip):
            pre_step_ball_vz = float(self._ball_vel()[2])
            mujoco.mj_step(self.model, self.data)
            self._scan_contacts(pre_step_ball_vz)

        self._update_landing()
        self.step_count += 1
        obs = self._get_obs()
        base_reward = self._compute_reward(applied)
        dist = float(np.linalg.norm(self._ball_pos() - self.target_pos))
        self.best_dist = min(self.best_dist, dist)
        landing_error = (
            None
            if self.landing_pos is None
            else float(np.linalg.norm(self.landing_pos[:2] - self.target_pos[:2]))
        )
        torso_height = float(self.data.xpos[self.torso_body_id, 2])
        torso_tilt_deg = self._torso_tilt_deg()
        has_fallen = bool(torso_height < 0.60 or torso_tilt_deg > 45.0)
        terminated = bool(self.landed or has_fallen or self._ball_pos()[0] > 4.0)
        truncated = bool(self.step_count * self.control_dt >= self.episode_time)
        self.observed_episode_max_tilt_deg = max(
            self.observed_episode_max_tilt_deg, torso_tilt_deg
        )
        if not self.released:
            self.observed_hold_max_tilt_deg = max(
                self.observed_hold_max_tilt_deg, torso_tilt_deg
            )
        elif self.release_time is not None:
            self.observed_post_release_tilts.append(
                (float(self.data.time - self.release_time), torso_tilt_deg)
            )

        info = {
            "dist_to_target": dist,
            "best_dist": float(self.best_dist),
            "released": self.released,
            "release_time": self.release_time,
            "landed": self.landed,
            "landing_pos": None
            if self.landing_pos is None
            else self.landing_pos.copy(),
            "landing_error_xy": landing_error,
            "success": bool(
                landing_error is not None
                and landing_error <= self.success_radius
                and not has_fallen
            ),
            "success_radius": self.success_radius,
            "has_fallen": has_fallen,
            "torso_height_m": torso_height,
            "torso_tilt_deg": torso_tilt_deg,
        }
        self.prev_action = applied.copy()

        reward = 0.02 * float(base_reward)
        if self.released and not was_released:
            reward += 0.5
        if info["landed"]:
            error = float(info["landing_error_xy"])
            reward += 60.0 - 500.0 * min(error, 0.25)
            if info["success"]:
                reward += 10.0
        if info["has_fallen"]:
            reward -= 60.0
        if truncated and not self.released:
            reward -= 30.0
        reward -= 0.002 * float(np.dot(residual, residual))
        return obs, float(reward), terminated, truncated, info


def first_sustained_stability_time(
    samples: list[tuple[float, float]], control_dt: float
) -> float | None:
    count = max(1, int(round(SETTLED_WINDOW_S / control_dt)))
    for index in range(max(0, len(samples) - count + 1)):
        window = samples[index : index + count]
        if all(tilt <= DOCUMENT_TILT_DIAGNOSTIC_DEG for _, tilt in window):
            return float(window[0][0])
    return None


def run_episode(env, policy, policy_name: str, episode: int, seed: int) -> dict:
    wall_start = time.perf_counter()
    obs, _ = env.reset(seed=seed)
    terminated = truncated = False
    reward_sum = 0.0
    info = {}
    last_action = np.zeros(env.action_space.shape, dtype=np.float32)

    while not (terminated or truncated):
        if policy is None:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            action, _ = policy.predict(obs, deterministic=True)
        last_action = np.asarray(action, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(last_action)
        reward_sum += reward

    official_has_fallen = bool(info.get("has_fallen", False))
    first_contact = (
        None if env.landing_pos is None else env.landing_pos.copy()
    )
    post_contact_positions: list[np.ndarray] = []
    post_contact_speeds: list[float] = []
    diagnostic_fall = official_has_fallen

    if first_contact is not None:
        diagnostic_steps = int(round(POST_CONTACT_DIAGNOSTIC_S / env.control_dt))
        for _ in range(diagnostic_steps):
            obs, _, _, _, diagnostic_info = env.step(last_action)
            position = env._ball_pos().copy()
            velocity = env._ball_vel().copy()
            post_contact_positions.append(position)
            post_contact_speeds.append(float(np.linalg.norm(velocity)))
            diagnostic_fall = diagnostic_fall or bool(
                diagnostic_info["has_fallen"]
            )

    final_position = (
        None
        if not post_contact_positions
        else post_contact_positions[-1]
    )
    settling_samples = max(
        1, int(round(SETTLED_WINDOW_S / env.control_dt))
    )
    ball_settled = bool(
        len(post_contact_speeds) >= settling_samples
        and all(
            speed <= SETTLED_SPEED_MPS
            for speed in post_contact_speeds[-settling_samples:]
        )
    )
    time_to_stability = first_sustained_stability_time(
        env.observed_post_release_tilts, env.control_dt
    )
    release_window_ok = bool(
        env.release_time is not None
        and abs(float(env.release_time) - RELEASE_TIME_CONFIG_S)
        <= env.control_dt + 1e-12
    )
    no_invalid_flight_contact = not env.observed_invalid_flight_contacts
    first_contact_success = bool(info.get("success", False))
    diagnostic_recovery = bool(
        time_to_stability is not None and not diagnostic_fall
    )
    stage_results = {
        "stable_hold_document_2deg_diagnostic": bool(
            env.observed_hold_max_tilt_deg <= DOCUMENT_TILT_DIAGNOSTIC_DEG
        ),
        "release_event": release_window_ok,
        "ball_flight_no_invalid_contact": no_invalid_flight_contact,
        "first_floor_contact": first_contact_success,
        "post_release_recovery_document_2deg_diagnostic": diagnostic_recovery,
    }
    invalid_reasons = []
    if not env.released:
        invalid_reasons.append("no_release")
    if first_contact is None:
        invalid_reasons.append("no_floor_contact")
    if not no_invalid_flight_contact:
        invalid_reasons.append("invalid_ball_contact")
    if official_has_fallen:
        invalid_reasons.append("official_fall")

    return {
        "policy_name": policy_name,
        "episode_id": episode,
        "seed": seed,
        "checkpoint_step": "best" if policy is not None else 0,
        "success": bool(info.get("success", False)),
        "ball_first_contact_x_m": None
        if first_contact is None
        else float(first_contact[0]),
        "ball_first_contact_y_m": None
        if first_contact is None
        else float(first_contact[1]),
        "first_contact_error_m": info.get("landing_error_xy"),
        "ball_final_x_m": None
        if final_position is None
        else float(final_position[0]),
        "ball_final_y_m": None
        if final_position is None
        else float(final_position[1]),
        "final_rolling_error_m": None
        if final_position is None
        else float(np.linalg.norm(final_position[:2] - TARGET_XY_M)),
        "resting_position_drift_m": None
        if final_position is None or first_contact is None
        else float(np.linalg.norm(final_position[:2] - first_contact[:2])),
        "ball_settled_by_3s": ball_settled,
        "release_time_s": env.release_time,
        "hold_max_torso_tilt_deg": env.observed_hold_max_tilt_deg,
        "torso_tilt_at_release_deg": env.observed_release_tilt_deg,
        "episode_max_torso_tilt_deg": env.observed_episode_max_tilt_deg,
        "time_to_2deg_stability_s": time_to_stability,
        "official_fall_flag": official_has_fallen,
        "post_contact_diagnostic_fall_flag": diagnostic_fall,
        "impact_velocity_z_mps": env.observed_impact_velocity_z_mps,
        "impact_speed_downward_mps": None
        if env.observed_impact_velocity_z_mps is None
        else max(0.0, -env.observed_impact_velocity_z_mps),
        "peak_floor_normal_force_n": env.observed_peak_impact_force_n,
        "invalid_flight_contacts": ";".join(
            sorted(env.observed_invalid_flight_contacts)
        ),
        "stage_results_json": json.dumps(stage_results, sort_keys=True),
        "stepwise_all_document_diagnostics_pass": all(stage_results.values()),
        "invalid_flag": bool(invalid_reasons),
        "invalid_reason": ";".join(invalid_reasons),
        "reward_sum": reward_sum,
        "wall_clock_duration_s": time.perf_counter() - wall_start,
    }


def numeric_summary(rows: list[dict], field: str) -> dict:
    values = np.asarray(
        [float(row[field]) for row in rows if row[field] is not None],
        dtype=np.float64,
    )
    if values.size == 0:
        return {"mean": None, "std": None, "p90": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def summarize(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "official_fall_rate": float(
            np.mean([row["official_fall_flag"] for row in rows])
        ),
        "post_contact_diagnostic_fall_rate": float(
            np.mean(
                [row["post_contact_diagnostic_fall_flag"] for row in rows]
            )
        ),
        "invalid_rate": float(np.mean([row["invalid_flag"] for row in rows])),
        "document_stepwise_diagnostic_pass_rate": float(
            np.mean(
                [
                    row["stepwise_all_document_diagnostics_pass"]
                    for row in rows
                ]
            )
        ),
        "ball_settled_by_3s_rate": float(
            np.mean([row["ball_settled_by_3s"] for row in rows])
        ),
        "first_contact_error_m": numeric_summary(
            rows, "first_contact_error_m"
        ),
        "final_rolling_error_m": numeric_summary(
            rows, "final_rolling_error_m"
        ),
        "resting_position_drift_m": numeric_summary(
            rows, "resting_position_drift_m"
        ),
        "torso_tilt_at_release_deg": numeric_summary(
            rows, "torso_tilt_at_release_deg"
        ),
        "hold_max_torso_tilt_deg": numeric_summary(
            rows, "hold_max_torso_tilt_deg"
        ),
        "episode_max_torso_tilt_deg": numeric_summary(
            rows, "episode_max_torso_tilt_deg"
        ),
        "time_to_2deg_stability_s": numeric_summary(
            rows, "time_to_2deg_stability_s"
        ),
        "impact_speed_downward_mps": numeric_summary(
            rows, "impact_speed_downward_mps"
        ),
        "peak_floor_normal_force_n": numeric_summary(
            rows, "peak_floor_normal_force_n"
        ),
        "reward_sum": numeric_summary(rows, "reward_sum"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "outputs" / "models" / "selected" / "best" / "best_model.zip",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--joint-noise", type=float, default=0.08)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "extended_evaluation",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model.resolve()
    policy = PPO.load(model_path, device="cpu")
    env = ObservedPPOThrowEnv(
        residual_scale=0.2,
        extra_initial_joint_noise=args.joint_noise,
    )
    all_rows: list[dict] = []
    summaries = {}
    for name, selected_policy in (("baseline", None), ("ppo_best", policy)):
        rows = [
            run_episode(
                env,
                selected_policy,
                name,
                episode,
                args.seed + episode,
            )
            for episode in range(args.episodes)
        ]
        all_rows.extend(rows)
        summaries[name] = summarize(rows)
    env.close()

    fields = list(all_rows[0])
    with (args.output_dir / "extended_episode_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "protocol": {
            "authority": "frozen project contract",
            "task1_commit": TASK1_COMMIT,
            "target_xy_m": TARGET_XY_M.tolist(),
            "success_radius_m": SUCCESS_RADIUS_M,
            "release_time_config_s": RELEASE_TIME_CONFIG_S,
            "episode_time_s": 1.8,
            "post_contact_diagnostic_s": POST_CONTACT_DIAGNOSTIC_S,
            "episodes_per_policy": args.episodes,
            "seed_start": args.seed,
            "joint_noise_rad": args.joint_noise,
            "air_density_enabled": False,
            "ppo_model": str(model_path),
            "ppo_model_sha256": sha256(model_path),
            "robot_model": "Unitree G1 29DoF",
            "robot_model_sha256": sha256(ROOT / "assets" / "g1.xml"),
            "scene_sha256": sha256(ROOT / "assets" / "scene_throw.xml"),
            "python": sys.version,
            "platform": platform.platform(),
            "mujoco_version": mujoco.__version__,
        },
        "baseline": summaries["baseline"],
        "ppo_best": summaries["ppo_best"],
        "interpretation": {
            "headline_success_uses_frozen_contract": True,
            "two_degree_stage_fields_are_diagnostics_only": True,
            "post_contact_rollout_does_not_change_headline_success": True,
        },
    }
    (args.output_dir / "extended_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
