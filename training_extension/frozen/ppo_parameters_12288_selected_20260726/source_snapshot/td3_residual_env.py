"""Closed-loop TD3 residual environment around frozen CEM v17.

Zero RL action reproduces the accepted CEM controller.  The learned policy can
only make small corrections on explicitly safe axes; it cannot move the root,
twist shoulders/wrists, approach the hoop, or replace the physical success
test.
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .basketball_env import BasketballResidualEnv
from .optimize_direct import controller_action


HERE = Path(__file__).resolve().parent
FROZEN_STATE = (
    HERE
    / "cem_artifacts"
    / "selected"
    / "state.json"
)

ARM_WAIST_NAMES = [
    "right_shoulder_pitch_joint",
    "right_elbow_joint",
    "right_wrist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_elbow_joint",
    "left_wrist_pitch_joint",
    "waist_pitch_joint",
]
LEG_PAIRS = [
    ("left_hip_pitch_joint", "right_hip_pitch_joint"),
    ("left_knee_joint", "right_knee_joint"),
    ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
]
RL_ACTION_NAMES = (
    ARM_WAIST_NAMES
    + ["symmetric_hip_pitch", "symmetric_knee", "symmetric_ankle_pitch"]
    + ["release_timing"]
)


class TD3BasketballResidualEnv(gym.Env):
    """Safe feedback residual around the frozen successful trajectory."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        expert_state: str | Path = FROZEN_STATE,
        arm_correction_scale: float = 0.05,
        leg_correction_scale: float = 0.05,
        release_correction_scale: float = 0.05,
        reward_scale: float = 0.01,
        residual_filter_alpha: float = 0.08,
        recovery_step: int = 520,
    ):
        super().__init__()
        self.base = BasketballResidualEnv(
            curriculum_radius=0.10,
            set_shot_only=False,
        )
        self.model = self.base.model
        self.data = self.base.data
        state = json.loads(Path(expert_state).read_text(encoding="utf-8"))
        self.expert_parameters = np.asarray(
            state["best_parameters"], dtype=np.float64
        )
        self.arm_correction_scale = float(arm_correction_scale)
        self.leg_correction_scale = float(leg_correction_scale)
        self.release_correction_scale = float(release_correction_scale)
        self.reward_scale = float(reward_scale)
        self.residual_filter_alpha = float(residual_filter_alpha)
        self.recovery_step = int(recovery_step)

        self.base_action_index = {
            name: i for i, name in enumerate(self.base.control_joint_names)
        }
        self.arm_action_ids = np.asarray(
            [self.base_action_index[name] for name in ARM_WAIST_NAMES],
            dtype=np.int32,
        )
        self.leg_action_ids = np.asarray(
            [
                [self.base_action_index[left], self.base_action_index[right]]
                for left, right in LEG_PAIRS
            ],
            dtype=np.int32,
        )
        self.foot_body_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"
                ),
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"
                ),
            ],
            dtype=np.int32,
        )
        if np.any(self.foot_body_ids < 0):
            raise RuntimeError("Foot bodies required for slip diagnostics are missing")
        self.backboard_contype = int(
            self.model.geom_contype[self.base.backboard_geom_id]
        )
        self.backboard_conaffinity = int(
            self.model.geom_conaffinity[self.base.backboard_geom_id]
        )

        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(len(RL_ACTION_NAMES),), dtype=np.float32
        )
        extra = 2 * len(RL_ACTION_NAMES)
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(self.base.observation_space.shape[0] + extra,),
            dtype=np.float32,
        )
        self.previous_rl_action = np.zeros(len(RL_ACTION_NAMES), dtype=np.float64)
        self.previous_ctrl = np.zeros(self.model.nu, dtype=np.float64)
        self.previous_foot_xy = np.zeros((2, 2), dtype=np.float64)
        self.crossing_rewarded = False
        self.shot_success_latched = False

    @property
    def policy(self):
        return self.base.policy

    def _expert_action(self):
        return controller_action(self.base, self.expert_parameters).astype(
            np.float64
        )

    def _expert_features(self):
        expert = self._expert_action()
        features = [expert[i] for i in self.arm_action_ids]
        features.extend(
            0.5 * (expert[left] + expert[right])
            for left, right in self.leg_action_ids
        )
        features.append(expert[-1])
        return np.asarray(features, dtype=np.float32)

    def _augment_observation(self, observation):
        return np.concatenate(
            [
                np.asarray(observation, dtype=np.float32),
                self.previous_rl_action.astype(np.float32),
                self._expert_features(),
            ]
        ).astype(np.float32)

    def _combined_action(self, rl_action):
        rl_action = np.clip(np.asarray(rl_action, dtype=np.float64), -1.0, 1.0)
        expert = self._expert_action()
        step = self.policy.step_count

        # During walking, arm corrections are deliberately smaller.  During
        # shooting/recovery they can use the full safe residual range.
        arm_phase_scale = 0.25 if step < 300 else 1.0
        for action_i, base_i in enumerate(self.arm_action_ids):
            expert[base_i] += (
                self.arm_correction_scale
                * arm_phase_scale
                * rl_action[action_i]
            )

        # Bilateral leg corrections cannot create an extra unilateral step.
        # They are disabled after the recovery begins.
        if step <= 500:
            leg_phase_scale = 1.0 if step < 300 else 0.50
            leg_offset = len(ARM_WAIST_NAMES)
            for pair_i, (left_i, right_i) in enumerate(self.leg_action_ids):
                correction = (
                    self.leg_correction_scale
                    * leg_phase_scale
                    * rl_action[leg_offset + pair_i]
                )
                expert[left_i] += correction
                expert[right_i] += correction

        expert[-1] += self.release_correction_scale * rl_action[-1]
        return np.clip(expert, -1.0, 1.0).astype(np.float32)

    def reset(self, seed=None, options=None):
        self.model.geom_contype[
            self.base.backboard_geom_id
        ] = self.backboard_contype
        self.model.geom_conaffinity[
            self.base.backboard_geom_id
        ] = self.backboard_conaffinity
        observation, info = self.base.reset(seed=seed, options=options)
        self.previous_rl_action[:] = 0.0
        self.previous_ctrl[:] = self.data.ctrl
        self.previous_foot_xy[:] = self.data.xpos[self.foot_body_ids, :2]
        self.crossing_rewarded = False
        self.shot_success_latched = False
        info = dict(info)
        info["rl_action_names"] = RL_ACTION_NAMES
        return self._augment_observation(observation), info

    def step(self, action):
        raw_rl_action = np.clip(
            np.asarray(action, dtype=np.float64), -1.0, 1.0
        )
        rl_action = (
            (1.0 - self.residual_filter_alpha) * self.previous_rl_action
            + self.residual_filter_alpha * raw_rl_action
        )
        applied = self._combined_action(rl_action)
        observation, base_reward, base_terminated, truncated, info = self.base.step(
            applied
        )

        action_energy = float(np.mean(rl_action * rl_action))
        action_delta = float(
            np.mean((rl_action - self.previous_rl_action) ** 2)
        )
        ctrl_delta = float(np.mean((self.data.ctrl - self.previous_ctrl) ** 2))

        foot_xy = self.data.xpos[self.foot_body_ids, :2].copy()
        foot_speed = np.linalg.norm(
            (foot_xy - self.previous_foot_xy) / 0.02, axis=1
        )
        foot_height = self.data.xpos[self.foot_body_ids, 2]
        stance_mask = foot_height < 0.12
        foot_slip = float(
            np.mean(foot_speed[stance_mask]) if np.any(stance_mask) else 0.0
        )

        smooth_penalty = (
            20.0 * action_energy
            + 10.0 * action_delta
            + 0.001 * ctrl_delta
            + 0.10 * foot_slip
        )
        reward = float(base_reward - smooth_penalty)

        if info["crossed_hoop_plane"] and not self.crossing_rewarded:
            self.crossing_rewarded = True
            error = float(
                info["crossing_xy_error"]
                if info["crossing_xy_error"] is not None
                else 1.0
            )
            if info["success"]:
                self.shot_success_latched = True
                reward += 400.0 + 1200.0 * max(0.0, 0.10 - error)
                # The shot has already passed the physical scoring plane.
                # Disable only later board contact so the recovery phase can
                # continue without relabeling a direct shot as a bank shot.
                self.model.geom_contype[self.base.backboard_geom_id] = 0
                self.model.geom_conaffinity[self.base.backboard_geom_id] = 0
            else:
                reward -= 300.0 + 200.0 * error
        reward *= self.reward_scale

        # A successful shot keeps running through follow-through/recovery so
        # hand and foot smoothness remain learnable.  Physical failures stop.
        hard_failure = bool(
            info["has_fallen"]
            or (info["touched_backboard"] and not self.shot_success_latched)
            or (info["crossed_hoop_plane"] and not info["success"])
        )
        recovered_success = bool(
            self.shot_success_latched
            and self.policy.step_count >= self.recovery_step
        )
        terminated = bool(hard_failure or recovered_success)
        if base_terminated and not info["success"]:
            terminated = True

        info = dict(info)
        info.update(
            {
                "success": bool(self.shot_success_latched),
                "direct_shot": bool(self.shot_success_latched),
                "rl_action_energy": action_energy,
                "rl_action_delta": action_delta,
                "raw_rl_action_energy": float(
                    np.mean(raw_rl_action * raw_rl_action)
                ),
                "ctrl_delta": ctrl_delta,
                "foot_slip_mps": foot_slip,
                "smoothness_penalty": smooth_penalty,
                "expert_centered": True,
            }
        )
        self.previous_rl_action[:] = rl_action
        self.previous_ctrl[:] = self.data.ctrl
        self.previous_foot_xy[:] = foot_xy
        return (
            self._augment_observation(observation),
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):
        self.base.close()
