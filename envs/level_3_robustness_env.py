"""
Robustness wrapper for Level 3 BasketballResidualEnv.
Adds 7 domain randomization parameters for Sim2Real testing.
"""

from __future__ import annotations
import numpy as np
import mujoco
from .basketball_env import BasketballResidualEnv


class RobustBasketballEnv(BasketballResidualEnv):
    """Level 3 Sim2Real: walk + two-hand throw with domain randomization."""

    def __init__(
        self,
        residual_scale: float = 1.0,
        # ── Domain Randomization ──
        obs_noise: float = 0.0,
        joint_friction_range: tuple[float, float] | None = None,
        joint_damping_range: tuple[float, float] | None = None,
        floor_friction_range: tuple[float, float] | None = None,
        actuator_gain_range: tuple[float, float] | None = None,
        target_pos_noise: float = 0.0,
        contact_solref_range: tuple[float, float] | None = None,
        contact_solimp_range: tuple[float, float] | None = None,
        control_latency_steps: int = 0,
        enable_all: bool = False,
    ):
        super().__init__(residual_scale=residual_scale)

        # ── Lookup floor geom ──
        self._floor_geom_ids = [
            i for i in range(self.model.ngeom)
            if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
        ]

        # ── Baselines ──
        self._baseline_joint_frictionloss = self.model.dof_frictionloss.copy()
        self._baseline_joint_damping = self.model.dof_damping.copy()
        self._baseline_actuator_forcerange = self.model.actuator_forcerange.copy()
        self._baseline_solref = self.model.opt.o_solref.copy()
        self._baseline_solimp = self.model.opt.o_solimp.copy()
        if self._floor_geom_ids:
            self._baseline_floor_friction = self.model.geom_friction[self._floor_geom_ids[0]].copy()

        # ── Config ──
        self.obs_noise = obs_noise
        self.joint_friction_range = joint_friction_range
        self.joint_damping_range = joint_damping_range
        self.floor_friction_range = floor_friction_range
        self.actuator_gain_range = actuator_gain_range
        self.target_pos_noise = target_pos_noise
        self.contact_solref_range = contact_solref_range
        self.contact_solimp_range = contact_solimp_range
        self.control_latency_steps = control_latency_steps
        self._action_buffer = []
        self.current_randomization = {}

        if enable_all:
            self._enable_all()

    def _enable_all(self):
        if self.obs_noise == 0.0:
            self.obs_noise = 0.02
        if self.joint_friction_range is None:
            self.joint_friction_range = (0.7, 1.3)
        if self.joint_damping_range is None:
            self.joint_damping_range = (0.7, 1.3)
        if self.floor_friction_range is None:
            self.floor_friction_range = (0.5, 1.5)
        if self.actuator_gain_range is None:
            self.actuator_gain_range = (0.85, 1.0)
        if self.target_pos_noise == 0.0:
            self.target_pos_noise = 0.03
        if self.contact_solref_range is None:
            self.contact_solref_range = (0.5, 2.0)
        if self.contact_solimp_range is None:
            self.contact_solimp_range = (0.5, 2.0)
        if self.control_latency_steps == 0:
            self.control_latency_steps = 3

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.current_randomization = {}
        self._action_buffer = []

        if self.joint_friction_range is not None:
            s = np.random.uniform(*self.joint_friction_range)
            self.model.dof_frictionloss[:] = self._baseline_joint_frictionloss * s
            self.current_randomization["joint_friction"] = s

        if self.joint_damping_range is not None:
            s = np.random.uniform(*self.joint_damping_range)
            self.model.dof_damping[:] = self._baseline_joint_damping * s
            self.current_randomization["joint_damping"] = s

        if self.floor_friction_range is not None:
            s = np.random.uniform(*self.floor_friction_range)
            for gid in self._floor_geom_ids:
                self.model.geom_friction[gid, 0] *= s
            self.current_randomization["floor_friction"] = s

        if self.actuator_gain_range is not None:
            s = np.random.uniform(*self.actuator_gain_range)
            self.model.actuator_forcerange[:] = self._baseline_actuator_forcerange * s
            self.current_randomization["actuator_gain"] = s

        if self.contact_solref_range is not None:
            s = np.random.uniform(*self.contact_solref_range)
            self.model.opt.o_solref[0] = self._baseline_solref[0] * s
            self.current_randomization["contact_solref"] = s

        if self.contact_solimp_range is not None:
            s = np.random.uniform(*self.contact_solimp_range)
            self.model.opt.o_solimp[0] = self._baseline_solimp[0] * s
            self.current_randomization["contact_solimp"] = s

        if self.target_pos_noise > 0:
            offset = np.random.normal(0, self.target_pos_noise, size=3)
            offset[2] = 0.0
            self.target = self.target + offset
            self.current_randomization["target_offset"] = offset

        mujoco.mj_forward(self.model, self.data)
        return obs, info

    def _get_obs(self):
        obs = super()._get_obs()
        if self.obs_noise > 0:
            obs = obs + np.random.normal(0, self.obs_noise, size=obs.shape).astype(np.float32)
        return obs

    def step(self, action):
        if self.control_latency_steps > 0:
            self._action_buffer.append(action.copy())
            if len(self._action_buffer) > self.control_latency_steps:
                action = self._action_buffer.pop(0)
            else:
                action = np.zeros_like(action)

        return super().step(action)

    def get_summary(self):
        return self.current_randomization.copy()
