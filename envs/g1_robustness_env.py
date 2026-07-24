"""
g1_robustness_env.py — Sim2Real Robustness Environment
======================================================
Subclass of teammate's PPOThrowEnv. Adds domain randomization,
observation noise, and external disturbances.

Usage:
    from envs.g1_robustness_env import G1RobustnessEnv

    # Clean baseline
    env = G1RobustnessEnv()

    # All perturbations ON
    env = G1RobustnessEnv(enable_all=True)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from envs.ppo_throw_env import PPOThrowEnv


class G1RobustnessEnv(PPOThrowEnv):
    """PPOThrowEnv + configurable Sim2Real perturbations."""

    def __init__(
        self,
        residual_scale: float = 0.2,
        extra_initial_joint_noise: float = 0.0,

        # ── Observation Noise ──
        obs_noise: float = 0.0,

        # ── Domain Randomization ──
        ball_mass_range: tuple[float, float] | None = None,
        joint_friction_range: tuple[float, float] | None = None,
        joint_damping_range: tuple[float, float] | None = None,
        floor_friction_range: tuple[float, float] | None = None,
        actuator_gain_range: tuple[float, float] | None = None,
        target_pos_noise: float = 0.0,
        ball_size_range: tuple[float, float] | None = None,

        # ── Actuator Perturbations ──
        control_latency_steps: int = 0,
        action_noise: float = 0.0,

        # ── Contact Modelling ──
        contact_solref_range: tuple[float, float] | None = None,
        # Multiplier for global contact stiffness (solref[0] = timeconst).
        # None = no change. e.g. (0.5, 2.0) = 50%-200% stiffness.
        # Simulates: surface compliance, contact deformation.
        contact_solimp_range: tuple[float, float] | None = None,
        # Multiplier for global contact impedance (solimp[0] = dmin).
        # None = no change. e.g. (0.5, 2.0).
        # Simulates: surface softness, penetration depth variation.

        # ── External Disturbances ──
        push_probability: float = 0.0,
        push_force_range: tuple[float, float] = (-3.0, 3.0),

        # ── Convenience ──
        enable_all: bool = False,
    ):
        super().__init__(
            residual_scale=residual_scale,
            extra_initial_joint_noise=extra_initial_joint_noise,
        )

        # ── Additional geom / body lookups ──
        self._floor_geom_ids = [
            i for i in range(self.model.ngeom)
            if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
        ]

        # ── Baseline snapshots ──
        self._baseline_ball_mass = self.model.body_mass[self.ball_body_id]
        self._baseline_joint_frictionloss = self.model.dof_frictionloss.copy()
        self._baseline_joint_damping = self.model.dof_damping.copy()
        self._baseline_actuator_forcerange = self.model.actuator_forcerange.copy()
        self._baseline_ball_size = self.model.geom_size[self.ball_geom_id].copy()
        self._baseline_target_pos = self.target_pos.copy()
        self._baseline_solref = self.model.opt.o_solref.copy()
        self._baseline_solimp = self.model.opt.o_solimp.copy()
        if self._floor_geom_ids:
            self._baseline_floor_friction = self.model.geom_friction[
                self._floor_geom_ids[0]
            ].copy()

        # ── Store config ──
        self.obs_noise = obs_noise
        self.ball_mass_range = ball_mass_range
        self.joint_friction_range = joint_friction_range
        self.joint_damping_range = joint_damping_range
        self.floor_friction_range = floor_friction_range
        self.actuator_gain_range = actuator_gain_range
        self.target_pos_noise = target_pos_noise
        self.ball_size_range = ball_size_range
        self.control_latency_steps = control_latency_steps
        self.action_noise = action_noise
        self.push_probability = push_probability
        self.push_force_range = push_force_range
        self.contact_solref_range = contact_solref_range
        self.contact_solimp_range = contact_solimp_range

        self._action_buffer = []
        self.current_randomization = {}

        if enable_all:
            self._enable_all_defaults()

    def _enable_all_defaults(self):
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
        if self.control_latency_steps == 0:
            self.control_latency_steps = 3
        if self.contact_solref_range is None:
            self.contact_solref_range = (0.5, 2.0)
        if self.contact_solimp_range is None:
            self.contact_solimp_range = (0.5, 2.0)

    # ═══════════════════════════════════════════════════════════
    #  RESET
    # ═══════════════════════════════════════════════════════════

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.current_randomization = {}
        self._action_buffer = []

        # ── Ball mass ──
        if self.ball_mass_range is not None:
            v = np.random.uniform(*self.ball_mass_range)
            self.model.body_mass[self.ball_body_id] = v
            self.current_randomization["ball_mass"] = v

        # ── Ball size ──
        if self.ball_size_range is not None:
            v = np.random.uniform(*self.ball_size_range)
            self.model.geom_size[self.ball_geom_id] = v
            self.current_randomization["ball_size"] = v

        # ── Joint friction ──
        if self.joint_friction_range is not None:
            s = np.random.uniform(*self.joint_friction_range)
            self.model.dof_frictionloss[:] = self._baseline_joint_frictionloss * s
            self.current_randomization["joint_friction_scale"] = s

        # ── Joint damping ──
        if self.joint_damping_range is not None:
            s = np.random.uniform(*self.joint_damping_range)
            self.model.dof_damping[:] = self._baseline_joint_damping * s
            self.current_randomization["joint_damping_scale"] = s

        # ── Floor friction ──
        if self.floor_friction_range is not None:
            s = np.random.uniform(*self.floor_friction_range)
            for gid in self._floor_geom_ids:
                self.model.geom_friction[gid, 0] *= s
            self.current_randomization["floor_friction_scale"] = s

        # ── Contact stiffness (solref) ──
        if self.contact_solref_range is not None:
            s = np.random.uniform(*self.contact_solref_range)
            self.model.opt.o_solref[0] = self._baseline_solref[0] * s
            self.current_randomization["contact_solref_scale"] = s

        # ── Contact impedance (solimp) ──
        if self.contact_solimp_range is not None:
            s = np.random.uniform(*self.contact_solimp_range)
            self.model.opt.o_solimp[0] = self._baseline_solimp[0] * s
            self.current_randomization["contact_solimp_scale"] = s

        # ── Actuator gain ──
        if self.actuator_gain_range is not None:
            s = np.random.uniform(*self.actuator_gain_range)
            self.model.actuator_forcerange[:] = self._baseline_actuator_forcerange * s
            self.current_randomization["actuator_gain"] = s

        # ── Target position noise ──
        if self.target_pos_noise > 0:
            offset = np.random.normal(0, self.target_pos_noise, size=3)
            offset[2] = 0.0
            new_pos = self._baseline_target_pos + offset
            self.model.body_pos[self.target_body_id] = new_pos
            self.target_pos = new_pos.copy()
            self.current_randomization["target_offset"] = offset

        mujoco.mj_forward(self.model, self.data)
        return obs, info

    # ═══════════════════════════════════════════════════════════
    #  OBSERVATION
    # ═══════════════════════════════════════════════════════════

    def _get_obs(self):
        obs = super()._get_obs()
        if self.obs_noise > 0:
            noise = np.random.normal(0, self.obs_noise, size=obs.shape).astype(np.float32)
            obs = obs + noise
        return obs

    # ═══════════════════════════════════════════════════════════
    #  STEP
    # ═══════════════════════════════════════════════════════════

    def step(self, action):
        # ── Action noise ──
        if self.action_noise > 0:
            action = action + np.random.normal(0, self.action_noise, size=action.shape)

        # ── Control latency ──
        if self.control_latency_steps > 0:
            self._action_buffer.append(action.copy())
            if len(self._action_buffer) > self.control_latency_steps:
                action = self._action_buffer.pop(0)
            else:
                action = np.zeros_like(action)

        # ── External push ──
        if self.push_probability > 0 and self.released and not self.landed:
            if np.random.rand() < self.push_probability:
                force = np.random.uniform(*self.push_force_range, size=3)
                self.data.xfrc_applied[self.ball_body_id, :3] += force

        return super().step(action)

    # ═══════════════════════════════════════════════════════════
    #  UTILITY
    # ═══════════════════════════════════════════════════════════

    def restore_baseline(self):
        self.model.body_mass[self.ball_body_id] = self._baseline_ball_mass
        self.model.dof_frictionloss[:] = self._baseline_joint_frictionloss
        self.model.opt.o_solref[:] = self._baseline_solref
        self.model.opt.o_solimp[:] = self._baseline_solimp
        self.model.dof_damping[:] = self._baseline_joint_damping
        self.model.actuator_forcerange[:] = self._baseline_actuator_forcerange
        self.model.geom_size[self.ball_geom_id] = self._baseline_ball_size
        self.target_pos = self._baseline_target_pos.copy()
        self.model.body_pos[self.target_body_id] = self._baseline_target_pos
        if self._floor_geom_ids:
            for gid in self._floor_geom_ids:
                self.model.geom_friction[gid, :] = self._baseline_floor_friction
        mujoco.mj_forward(self.model, self.data)

    def get_randomization_summary(self) -> dict:
        return self.current_randomization.copy()

    def print_randomization(self):
        print("-" * 40)
        print("  Sim2Real Randomization")
        print("-" * 40)
        for key, val in sorted(self.current_randomization.items()):
            if isinstance(val, np.ndarray):
                print(f"  {key}: {np.array2string(val, precision=4)}")
            elif isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")
        if not self.current_randomization:
            print("  (none — clean baseline)")
        print("-" * 40)
