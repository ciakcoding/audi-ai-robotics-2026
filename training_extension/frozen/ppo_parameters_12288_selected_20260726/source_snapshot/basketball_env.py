from __future__ import annotations

import importlib.util
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .derived_baseline import PeerEnv, TrainingBasketballPolicy


ROOT = Path(__file__).resolve().parents[1]
BASELINE_SOURCE = ROOT / "scripts" / "view_baselines_LEVEL03_v030!.py"
SCENE = ROOT / "training_extension" / "scene_throw_LEVEL03_ring.xml"


class BasketballResidualEnv(gym.Env):
    """Physical hoop environment around the derived scripted baseline.

    A zero action executes the baseline exactly.  The bounded action interface
    is kept so the stacked optimization branch can reuse the same immutable
    world and scoring contract.
    """

    metadata = {"render_modes": []}
    target = np.array([2.2, 0.0, 1.2], dtype=np.float64)
    hoop_radius = 0.10
    # The peer viewer stops at 850 while the ball is still above the hoop.
    # Keep its controls unchanged, but allow the physical flight to finish.
    max_policy_steps = 1100
    release_policy_step = 406

    def __init__(
        self,
        residual_scale: float = 1.0,
        curriculum_radius: float = 0.10,
        set_shot_only: bool = False,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.curriculum_radius = float(curriculum_radius)
        self.set_shot_only = bool(set_shot_only)
        if self.curriculum_radius < self.hoop_radius:
            raise ValueError("curriculum_radius cannot be smaller than 0.10 m")
        self.peer_env = PeerEnv(xml_path=SCENE)
        self.model = self.peer_env.model
        self.data = self.peer_env.data
        self.policy = TrainingBasketballPolicy(self.peer_env)
        self.pelvis_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        self.ball_id = self.peer_env.ball_body_id
        self.hand_body_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"
                ),
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"
                ),
            ],
            dtype=np.int32,
        )
        self.ball_dof_adr = self.peer_env.ball_qvel_adr
        self.backboard_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "backboard"
        )
        self.control_substeps = self.peer_env.frame_skip
        self.actuated_joint_qpos = np.array(
            [
                self.model.jnt_qposadr[int(self.model.actuator_trnid[i, 0])]
                for i in range(self.model.nu)
            ],
            dtype=np.int32,
        )
        self.actuated_joint_qvel = np.array(
            [
                self.model.jnt_dofadr[int(self.model.actuator_trnid[i, 0])]
                for i in range(self.model.nu)
            ],
            dtype=np.int32,
        )
        self.control_joint_names = [
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
            "right_wrist_roll_joint", "right_wrist_pitch_joint",
            "right_wrist_yaw_joint", "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint",
            "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_hip_pitch_joint", "left_knee_joint",
            "left_ankle_pitch_joint", "right_hip_pitch_joint",
            "right_knee_joint", "right_ankle_pitch_joint",
        ]
        missing = [
            name for name in self.control_joint_names
            if name not in self.policy.actuator_map
        ]
        if missing:
            raise RuntimeError(f"Missing controlled actuators: {missing}")
        self.residual_actuator_ids = np.array(
            [self.policy.actuator_map[name] for name in self.control_joint_names],
            dtype=np.int32,
        )
        # Radians added to the peer trajectory at full action magnitude.
        self.joint_residual_scales = np.array(
            [2.00] * 14 + [0.35] * 3 + [0.45] * 6, dtype=np.float64
        )
        self.rim_geom_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, f"rim_{i:02d}"
                )
                for i in range(16)
            ],
            dtype=np.int32,
        )
        if np.any(self.rim_geom_ids < 0):
            raise RuntimeError("The 16-segment physical rim is incomplete")
        # Wide curriculum gates are virtual. The immutable 10 cm evaluation
        # always restores physical ring collisions.
        if self.curriculum_radius > self.hoop_radius:
            self.model.geom_contype[self.rim_geom_ids] = 0
            self.model.geom_conaffinity[self.rim_geom_ids] = 0
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(len(self.residual_actuator_ids) + 1,),
            dtype=np.float32,
        )
        obs_size = self.model.nu * 2 + 3 + 3 + 3 + 4 + 3 + 2
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.released = False
        self.crossed_plane = False
        self.success = False
        self.previous_ball_pos = np.zeros(3)
        self.previous_distance = 0.0
        self.closest_hoop_xy_error = np.inf
        self.previous_predicted_crossing_error = 1.0
        self.touched_backboard = False
        self.filtered_residual = np.zeros(len(self.residual_actuator_ids))
        self.previous_residual = np.zeros(len(self.residual_actuator_ids))
        self.release_step = None
        self.release_ball_pos = None
        self.release_ball_velocity = None
        self.release_pelvis_pos = None
        self.release_hand_separation = None
        self.crossing_ball_pos = None
        self.crossing_xy_error = None
        self.minimum_hand_to_hoop_distance = np.inf
        self.airborne_horizontal_distance = 0.0

    @staticmethod
    def _tilt_degrees(quat):
        w, x, y, z = quat
        sinr = 2 * (w * x + y * z)
        cosr = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr, cosr)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
        siny = 2 * (w * z + x * y)
        cosy = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny, cosy)
        return np.degrees([pitch, roll, yaw])

    def _apply_peer_stabilizer(self):
        pitch, roll, yaw = self._tilt_degrees(self.data.xquat[self.pelvis_id])
        self.data.xfrc_applied[:] = 0.0
        self.data.xfrc_applied[self.pelvis_id, 3] = np.clip(
            -roll * 100.0 - self.data.qvel[3] * 20.0, -200.0, 200.0
        )
        self.data.xfrc_applied[self.pelvis_id, 4] = np.clip(
            -pitch * 100.0 - self.data.qvel[4] * 20.0, -200.0, 200.0
        )
        self.data.xfrc_applied[self.pelvis_id, 5] = np.clip(
            -yaw * 50.0 - self.data.qvel[5] * 10.0, -100.0, 100.0
        )
        self.data.xfrc_applied[self.pelvis_id, 1] = np.clip(
            -self.data.qpos[1] * 50.0 - self.data.qvel[1] * 10.0,
            -50.0,
            50.0,
        )

    def _ball_velocity(self):
        return self.data.qvel[self.ball_dof_adr : self.ball_dof_adr + 3].copy()

    def _get_obs(self):
        ball_pos = self.data.xpos[self.ball_id]
        phase = self.policy.step_count / self.max_policy_steps
        return np.concatenate(
            [
                self.data.qpos[self.actuated_joint_qpos],
                self.data.qvel[self.actuated_joint_qvel],
                ball_pos,
                self._ball_velocity(),
                self.data.xpos[self.pelvis_id],
                self.data.xquat[self.pelvis_id],
                self.target - ball_pos,
                [phase, float(self.released)],
            ]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.peer_env.reset(seed=seed)
        self.policy.reset()
        if self.set_shot_only:
            # Start at the peer baseline's square-up/dip sequence instead of
            # walking all the way to a hoop only 1.8 m from the initial pelvis.
            self.policy.step_count = 350
        self.peer_env.model.body_pos[self.peer_env.target_body_id] = self.target
        self.data.qpos[:3] = [0.0, 0.0, 0.81]
        self.data.qvel[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.released = False
        self.crossed_plane = False
        self.success = False
        self.previous_ball_pos = self.data.xpos[self.ball_id].copy()
        self.previous_distance = float(np.linalg.norm(self.previous_ball_pos - self.target))
        self.closest_hoop_xy_error = np.inf
        self.previous_predicted_crossing_error = 1.0
        self.touched_backboard = False
        self.filtered_residual[:] = 0.0
        self.previous_residual[:] = 0.0
        self.release_step = None
        self.release_ball_pos = None
        self.release_ball_velocity = None
        self.release_pelvis_pos = None
        self.release_hand_separation = None
        self.crossing_ball_pos = None
        self.crossing_xy_error = None
        self.minimum_hand_to_hoop_distance = np.inf
        self.airborne_horizontal_distance = 0.0
        return self._get_obs(), self._info()

    def _predicted_crossing_error(self):
        """Ballistic XY error where the released ball should cross hoop height."""
        pos = self.data.xpos[self.ball_id]
        vel = self._ball_velocity()
        height = float(pos[2] - self.target[2])
        # Solve height + vz*t - 0.5*g*t^2 = 0 for the future positive root.
        gravity = abs(float(self.model.opt.gravity[2]))
        discriminant = vel[2] * vel[2] + 2.0 * gravity * height
        if discriminant < 0.0:
            return None
        flight_time = (vel[2] + np.sqrt(discriminant)) / gravity
        if flight_time <= 0.0 or flight_time > 2.0:
            return None
        predicted_xy = pos[:2] + vel[:2] * flight_time
        return float(np.linalg.norm(predicted_xy - self.target[:2]))

    def _info(self):
        ball_pos = self.data.xpos[self.ball_id].copy()
        pitch, roll, yaw = self._tilt_degrees(self.data.xquat[self.pelvis_id])
        pelvis_height = float(self.data.xpos[self.pelvis_id, 2])
        fallen = pelvis_height < 0.45 or max(abs(pitch), abs(roll)) > 60.0
        return {
            "success": bool(self.success),
            "curriculum_radius": self.curriculum_radius,
            "crossed_hoop_plane": bool(self.crossed_plane),
            "hoop_xy_error": float(
                self.closest_hoop_xy_error
                if np.isfinite(self.closest_hoop_xy_error)
                else 1.0
            ),
            "released": bool(self.released),
            "release_step": self.release_step,
            "release_distance_to_hoop_xy": (
                None
                if self.release_ball_pos is None
                else float(
                    np.linalg.norm(self.release_ball_pos[:2] - self.target[:2])
                )
            ),
            "release_ball_position": (
                None
                if self.release_ball_pos is None
                else self.release_ball_pos.tolist()
            ),
            "release_ball_velocity": (
                None
                if self.release_ball_velocity is None
                else self.release_ball_velocity.tolist()
            ),
            "release_pelvis_position": (
                None
                if self.release_pelvis_pos is None
                else self.release_pelvis_pos.tolist()
            ),
            "release_pelvis_distance_to_hoop_xy": (
                None
                if self.release_pelvis_pos is None
                else float(
                    np.linalg.norm(self.release_pelvis_pos[:2] - self.target[:2])
                )
            ),
            "release_hand_separation": self.release_hand_separation,
            "crossing_ball_position": (
                None
                if self.crossing_ball_pos is None
                else self.crossing_ball_pos.tolist()
            ),
            "crossing_xy_error": self.crossing_xy_error,
            "airborne_horizontal_distance": float(
                self.airborne_horizontal_distance
            ),
            "minimum_hand_to_hoop_distance": float(
                self.minimum_hand_to_hoop_distance
            ),
            "touched_backboard": bool(self.touched_backboard),
            "direct_shot": bool(self.success and not self.touched_backboard),
            "policy_step": int(self.policy.step_count),
            "has_fallen": bool(fallen),
            "pelvis_height_m": pelvis_height,
            "pitch_deg": float(pitch),
            "roll_deg": float(roll),
            "yaw_deg": float(yaw),
        }

    def step(self, action):
        full_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if full_action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {full_action.shape}"
            )
        residual = full_action[: len(self.residual_actuator_ids)]
        release_signal = float(full_action[-1])
        just_released = False

        self.policy.apply_controls()
        # Preserve the peer baseline exactly at action=0 while allowing a
        # continuous +/-24 policy-step timing adjustment.
        desired_release_step = int(np.rint(406.0 + 24.0 * release_signal))
        if (
            self.policy.step_count >= desired_release_step
            or self.policy.step_count >= 430
        ) and not self.released:
            self.data.eq_active[self.peer_env.hold_eq_id] = 0
            self.released = True
            just_released = True
            self.release_step = int(self.policy.step_count)
            self.release_ball_pos = self.data.xpos[self.ball_id].copy()
            self.release_ball_velocity = self._ball_velocity()
            self.release_pelvis_pos = self.data.xpos[self.pelvis_id].copy()
            self.release_hand_separation = float(
                np.linalg.norm(
                    self.data.xpos[self.hand_body_ids[0]]
                    - self.data.xpos[self.hand_body_ids[1]]
                )
            )

        ids = self.residual_actuator_ids
        residual_active = 300 <= self.policy.step_count <= 500
        if residual_active:
            self.filtered_residual = 0.85 * self.filtered_residual + 0.15 * residual
            self.data.ctrl[ids] = np.clip(
                self.data.ctrl[ids]
                + self.residual_scale
                * self.joint_residual_scales
                * self.filtered_residual,
                self.model.actuator_ctrlrange[ids, 0],
                self.model.actuator_ctrlrange[ids, 1],
            )
        crossing_xy_error = None
        new_backboard_contact = False
        substep_previous_pos = self.data.xpos[self.ball_id].copy()
        for _ in range(self.control_substeps):
            self._apply_peer_stabilizer()
            mujoco.mj_step(self.model, self.data)
            ball_pos = self.data.xpos[self.ball_id].copy()
            hand_distance = min(
                float(
                    np.linalg.norm(
                        self.data.xpos[hand_id] - self.target
                    )
                )
                for hand_id in self.hand_body_ids
            )
            self.minimum_hand_to_hoop_distance = min(
                self.minimum_hand_to_hoop_distance, hand_distance
            )
            if self.released and not self.touched_backboard:
                for contact_index in range(self.data.ncon):
                    contact = self.data.contact[contact_index]
                    pair = {int(contact.geom1), int(contact.geom2)}
                    if pair == {self.peer_env.ball_geom_id, self.backboard_geom_id}:
                        self.touched_backboard = True
                        new_backboard_contact = True
                        break
            if (
                self.released
                and not self.crossed_plane
                and substep_previous_pos[2] > self.target[2]
                and ball_pos[2] <= self.target[2]
                and self._ball_velocity()[2] < 0.0
            ):
                self.crossed_plane = True
                self.crossing_ball_pos = ball_pos.copy()
                crossing_xy_error = float(
                    np.linalg.norm(ball_pos[:2] - self.target[:2])
                )
                self.crossing_xy_error = crossing_xy_error
                self.closest_hoop_xy_error = min(
                    self.closest_hoop_xy_error, crossing_xy_error
                )
                self.airborne_horizontal_distance = float(
                    np.linalg.norm(ball_pos[:2] - self.release_ball_pos[:2])
                )
                release_distance = float(
                    np.linalg.norm(self.release_ball_pos[:2] - self.target[:2])
                )
                self.success = (
                    crossing_xy_error <= self.curriculum_radius
                    and not self.touched_backboard
                    and release_distance >= 1.10
                    and float(
                        np.linalg.norm(
                            self.release_pelvis_pos[:2] - self.target[:2]
                        )
                    ) >= 1.20
                    and self.release_ball_pos[2] >= 1.20
                    and self.release_hand_separation <= 0.25
                    and self.airborne_horizontal_distance >= 1.00
                    and self.minimum_hand_to_hoop_distance >= 0.45
                )
                break
            substep_previous_pos = ball_pos

        ball_pos = self.data.xpos[self.ball_id].copy()
        distance = float(np.linalg.norm(ball_pos - self.target))
        if self.released and abs(ball_pos[2] - self.target[2]) <= 0.15:
            self.closest_hoop_xy_error = min(
                self.closest_hoop_xy_error,
                float(np.linalg.norm(ball_pos[:2] - self.target[:2])),
            )
        reward = 0.0
        if just_released:
            release_velocity = self._ball_velocity()
            gravity = abs(float(self.model.opt.gravity[2]))
            ballistic_apex = float(
                ball_pos[2]
                + max(float(release_velocity[2]), 0.0) ** 2 / (2.0 * gravity)
            )
            reward += 20.0 * float(release_velocity[0])
            reward -= 20.0 * abs(float(release_velocity[1]))
            reward -= 120.0 * max(self.target[2] - ballistic_apex, 0.0)
        if residual_active:
            reward -= 0.001 * float(
                np.dot(self.filtered_residual, self.filtered_residual)
            )
            reward -= 0.01 * float(
                np.dot(
                    self.filtered_residual - self.previous_residual,
                    self.filtered_residual - self.previous_residual,
                )
            )
            pitch, roll, _ = self._tilt_degrees(self.data.xquat[self.pelvis_id])
            reward -= 0.0002 * float(
                pitch * pitch + roll * roll
                + 25.0 * self.data.qpos[1] * self.data.qpos[1]
            )
        predicted_error = self._predicted_crossing_error()
        if predicted_error is not None:
            # Potential-based shaping: reward improvement in the predicted
            # crossing point, not merely getting spatially close to the rim.
            reward += float(
                np.clip(
                    400.0
                    * (self.previous_predicted_crossing_error - predicted_error),
                    -30.0,
                    30.0,
                )
            )
            self.previous_predicted_crossing_error = predicted_error

        if crossing_xy_error is not None:
            xy_error = crossing_xy_error
            # Dense accuracy score remains informative outside the current
            # gate; the success bonus is still tied to that stage's radius.
            reward += 250.0 * float(np.exp(-xy_error / 0.20))
            if self.success:
                reward += 300.0 + 1000.0 * (
                    self.curriculum_radius - xy_error
                )
            else:
                reward -= 50.0 * xy_error
        if new_backboard_contact:
            reward -= 150.0

        info = self._info()
        if info["has_fallen"]:
            reward -= 100.0
        terminated = bool(
            self.crossed_plane or self.touched_backboard or info["has_fallen"]
        )
        truncated = bool(self.policy.step_count >= self.max_policy_steps)
        if truncated and not self.success:
            reward -= 30.0

        self.previous_ball_pos = ball_pos
        self.previous_distance = distance
        self.previous_residual[:] = self.filtered_residual
        return self._get_obs(), float(reward), terminated, truncated, info

    def close(self):
        self.peer_env.close()
