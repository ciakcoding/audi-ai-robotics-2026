"""One-decision SAC environment around the frozen CEM v17 trajectory.

The learned action is a small *parameter residual*, not a per-frame joint
command.  The accepted walk, dip, two-hand release and separated recovery are
therefore preserved, while SAC can robustify launch parameters and timing
against the arm-pose perturbation already present in the teammate environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .basketball_env import BasketballResidualEnv
from .optimize_direct import PARAMETER_NAMES, controller_action
from .td3_residual_env import FROZEN_STATE


class SACShotParameterEnv(gym.Env):
    """Contextual, episodic SAC task with immutable physical scoring."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        expert_state: str | Path = FROZEN_STATE,
        parameter_scale: float = 1.0,
        timing_scale: float = 0.0,
    ):
        super().__init__()
        self.base = BasketballResidualEnv(
            curriculum_radius=0.10,
            set_shot_only=False,
        )
        state = json.loads(Path(expert_state).read_text(encoding="utf-8"))
        self.expert_parameters = np.asarray(
            state["best_parameters"], dtype=np.float64
        )
        if self.expert_parameters.shape != (len(PARAMETER_NAMES),):
            raise RuntimeError("Frozen CEM parameter contract changed")
        # Per-axis limits come from a one-at-a-time physical sensitivity
        # probe.  Load elbows and guide-elbow release are particularly
        # sensitive; timing is frozen because a one-step shift can bank-shot.
        self.parameter_scales = float(parameter_scale) * np.asarray(
            [
                0.005, 0.002, 0.005, 0.005, 0.002, 0.005, 0.010,
                0.010, 0.005, 0.005, 0.005, 0.002, 0.005, 0.010,
                float(timing_scale),
            ],
            dtype=np.float64,
        )
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(len(PARAMETER_NAMES),),
            dtype=np.float32,
        )
        # Initial physical state supplies the context for the randomized arm
        # pose.  The frozen parameters make the residual convention explicit.
        obs_size = self.base.observation_space.shape[0] + len(PARAMETER_NAMES)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_size,), dtype=np.float32
        )
        self._initial_observation = None

    def _observation(self):
        return np.concatenate(
            [
                np.asarray(self._initial_observation, dtype=np.float32),
                self.expert_parameters.astype(np.float32),
            ]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        observation, info = self.base.reset(seed=seed, options=options)
        self._initial_observation = observation.copy()
        info = dict(info)
        info.update(
            {
                "parameter_names": PARAMETER_NAMES,
                "expert_centered": True,
            }
        )
        return self._observation(), info

    def step(self, action):
        residual = np.clip(
            np.asarray(action, dtype=np.float64), -1.0, 1.0
        )
        parameters = self.expert_parameters + self.parameter_scales * residual
        terminated = truncated = False
        info = {}
        control_deltas = []
        previous_ctrl = self.base.data.ctrl.copy()
        while not (terminated or truncated):
            full_action = controller_action(self.base, parameters)
            _, _, terminated, truncated, info = self.base.step(full_action)
            control_deltas.append(
                float(np.mean((self.base.data.ctrl - previous_ctrl) ** 2))
            )
            previous_ctrl[:] = self.base.data.ctrl

        crossing_error = float(
            info["crossing_xy_error"]
            if info["crossing_xy_error"] is not None
            else info["hoop_xy_error"]
        )
        success = bool(info["success"])
        # Reward stays numerically compact for SAC while strongly preferring
        # legal direct makes.  Accuracy remains informative on both sides of
        # the fixed 10 cm boundary.
        reward = (
            10.0 * float(success)
            + 2.0 * float(np.exp(-crossing_error / 0.03))
            - 10.0 * crossing_error
            - 0.20 * float(np.mean(residual * residual))
        )
        if not info["crossed_hoop_plane"]:
            reward -= 5.0
        if info["touched_backboard"]:
            reward -= 10.0
        if info["has_fallen"]:
            reward -= 20.0
        if info["minimum_hand_to_hoop_distance"] < 0.45:
            reward -= 10.0
        if info["airborne_horizontal_distance"] < 1.00:
            reward -= 10.0

        info = dict(info)
        info.update(
            {
                "success": success,
                "direct_shot": bool(
                    success and not info["touched_backboard"]
                ),
                "parameter_residual": residual.astype(float).tolist(),
                "parameter_l2": float(np.mean(residual * residual)),
                "mean_ctrl_delta": float(np.mean(control_deltas)),
                "expert_centered": True,
            }
        )
        # A single RL transition represents the complete physical shot.
        return self._observation(), float(reward), True, False, info

    def close(self):
        self.base.close()
