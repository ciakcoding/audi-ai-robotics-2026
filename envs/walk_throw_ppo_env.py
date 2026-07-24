"""
walk_throw_ppo_env.py — Level 3: Residual PPO for walk + throw
===============================================================
Baseline: PD stands still, arm scripted throw.
PPO learns residual corrections to walk forward while throwing.

Same pattern as ppo_throw_env.py (Level 0/1 teammate version).
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import mujoco
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from envs.g1_walk_throw_env import G1WalkThrowEnv


class WalkThrowPPOEnv(G1WalkThrowEnv):
    """Residual PPO: baseline stands + throws, PPO learns to walk."""

    def __init__(
        self,
        residual_scale: float = 1.0,
        extra_initial_joint_noise: float = 0.0,
    ):
        super().__init__()

        self.residual_scale = float(residual_scale)
        self.extra_initial_joint_noise = float(extra_initial_joint_noise)

        # PPO only outputs residuals, no release action (scripted release)
        self.action_space = spaces.Box(
            -1, 1, shape=(self.n_controlled,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        if self.extra_initial_joint_noise > 0:
            self.data.qpos[self.controlled_qpos_adr] += self.np_random.uniform(
                -self.extra_initial_joint_noise,
                self.extra_initial_joint_noise,
                self.n_controlled,
            )
            mujoco.mj_forward(self.model, self.data)
            self._place_ball_in_hand()
            mujoco.mj_forward(self.model, self.data)
            obs = self._get_obs()

        return obs, info

    def step(self, action):
        # Residual action: PPO correction on top of baseline
        was_released = self.released
        residual = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # Get baseline action (stand still + scripted arm throw)
        t = self.step_count * self.control_dt
        baseline = self.get_baseline_action(t)

        # Apply baseline + residual correction
        applied = np.clip(baseline + self.residual_scale * residual, -1.0, 1.0)

        obs, base_reward, terminated, truncated, info = super().step(applied)

        # ── Residual reward shaping ──
        reward = 0.02 * float(base_reward)

        # Ball release bonus
        if self.released and not was_released:
            reward += 0.5

        # Landing precision
        if info["landed"]:
            error = float(info.get("landing_error_xy", 0.5))
            reward += 60.0 - 500.0 * min(error, 0.25)
            if info["success"]:
                reward += 10.0

        # Fall penalty
        if info["has_fallen"]:
            reward -= 60.0

        # No release penalty
        if terminated and not self.released:
            reward -= 30.0
        if truncated and not self.released:
            reward -= 30.0

        # Forward progress bonus (walking!)
        pelvis_vx = float(self.data.qvel[0])
        if not info["has_fallen"]:
            reward += 2.0 * max(0.0, pelvis_vx)  # reward forward velocity

        # Regularization
        reward -= 0.002 * float(np.dot(residual, residual))

        info["task2_base_reward"] = float(base_reward)
        info["task2_reward"] = float(reward)
        info["task2_residual_l2"] = float(np.linalg.norm(residual))

        return obs, reward, terminated, truncated, info
