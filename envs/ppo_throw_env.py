"""Task 2 learning wrapper around the frozen Task 1 world model."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces


PROJECT_ROOT = Path(__file__).resolve().parents[1]

from envs.g1_fixed_body_throw_env import G1FixedBodyThrowEnv


class PPOThrowEnv(G1FixedBodyThrowEnv):
    """Residual PPO around the frozen, disclosed Task 1 baseline."""

    def __init__(
        self,
        residual_scale: float = 0.2,
        extra_initial_joint_noise: float = 0.0,
    ):
        super().__init__(
            xml_path=PROJECT_ROOT / "assets" / "scene_throw.xml",
            learned_release=False,
        )
        self.residual_scale = float(residual_scale)
        self.extra_initial_joint_noise = float(extra_initial_joint_noise)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.baseline_start = np.array(
            [0.9318, -0.7911, 0.0491, -0.1425, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self.baseline_end = np.array(
            [-1.0, 0.0964, 0.0072, -1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.extra_initial_joint_noise > 0:
            self.data.qpos[self.arm_qpos_adr] += self.np_random.uniform(
                -self.extra_initial_joint_noise,
                self.extra_initial_joint_noise,
                self.n_arm,
            )
            mujoco.mj_forward(self.model, self.data)
            self._place_ball_in_hand()
            mujoco.mj_forward(self.model, self.data)
            obs = self._get_obs()
        info["task2_extra_initial_joint_noise"] = self.extra_initial_joint_noise
        return obs, info

    def step(self, action):
        was_released = self.released
        residual = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        progress = min(1.0, self.step_count / 34.0)
        baseline = self.baseline_start + progress * (
            self.baseline_end - self.baseline_start
        )
        applied = np.zeros(8, dtype=np.float64)
        applied[:7] = np.clip(
            baseline + self.residual_scale * residual,
            -1.0,
            1.0,
        )
        obs, base_reward, terminated, truncated, info = super().step(applied)
        reward = 0.02 * float(base_reward)

        # Reward guides learning only; final success still comes unchanged
        # from the frozen Task 1 first-contact and stability metrics.
        if self.released and not was_released:
            reward += 0.5
        if info["landed"]:
            error = float(info["landing_error_xy"])
            # Continuous precision term: being barely inside the success
            # radius is not equivalent to landing at the target center.
            reward += 60.0 - 500.0 * min(error, 0.25)
            if info["success"]:
                reward += 10.0
        if info["has_fallen"]:
            reward -= 60.0
        if truncated and not self.released:
            reward -= 30.0
        reward -= 0.002 * float(np.dot(residual, residual))

        info["task2_base_reward"] = float(base_reward)
        info["task2_reward"] = float(reward)
        info["task2_residual_l2"] = float(np.linalg.norm(residual))
        return obs, reward, terminated, truncated, info
