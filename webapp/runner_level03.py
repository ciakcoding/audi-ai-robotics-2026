"""Runs the Level 03 comparison headlessly and streams it: the scripted
basketball baseline from scripts/view_baselines_LEVEL03_v031!.py on one
side, the trained RL policy from training_extension/view_ppo_parameters.py
on the other -- each using its own script's actual environment, scene, and
simulation loop exactly as written, not a shared env either script's
author didn't build. That means the two sides are two genuinely different
scenes (different target distance, different scene XML, different episode
length/success criteria) rendered side by side, not an apples-to-apples
lockstep comparison the way Level 02's baseline/RL pair is -- there is no
unified Level 03 env in this project to make that lockstep comparison
meaningful, so this module doesn't invent one.

Baseline side (BaselineSlot): dynamically loads
scripts/view_baselines_LEVEL03_v031!.py (importlib, same technique
training_extension/derived_baseline.py already uses for this exact file)
and replays its view_baseline() function's per-step body verbatim --
OptionDBasketballPolicy's keyframe controls, the pelvis anti-drift gyro
correction, the release-at-step-406 weld release, the hoop-crossing and
rim-impact-force detection -- just swapping the interactive
mujoco.viewer.launch_passive() loop for headless stepping + offscreen
rendering. Runs the same fixed 850 steps the script always runs (it has no
early-exit condition).

RL side (RLSlot): loads the same trained PPO model + VecNormalize
training_extension/view_ppo_parameters.py loads, over
training_extension/sac_parameter_env.py's SACShotParameterEnv (one-decision
parameter residual, not a per-step action -- see that module's docstring).
Terminates early (~426 steps) per BasketballResidualEnv's own scoring
contract, then holds a stabilized idle pose (via
BasketballResidualEnv._apply_peer_stabilizer(), the same anti-drift torques
its own step() applies every substep) for the remainder of the 850-step
window so it doesn't collapse once its episode ends, and so both panels
run for the same visual duration.

assets/g1.xml's meshdir was fixed (assets -> .) to make
assets/scene_throw_LEVEL03.xml loadable at all -- the same broken-mesh-path
bug worked around for Level 02's webapp, but this time there was no
existing self-contained alternative scene to point at instead, so the
shared asset itself was corrected.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Union

import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from training_extension.optimize_direct import controller_action
from training_extension.sac_parameter_env import SACShotParameterEnv

from webapp.runner import StreamFrame

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCRIPT_PATH = ROOT / "scripts" / "view_baselines_LEVEL03_v031!.py"
BASELINE_SCENE_PATH = ROOT / "assets" / "scene_throw_LEVEL03.xml"

FROZEN_DIR = ROOT / "training_extension" / "frozen" / "ppo_parameters_12288_selected_20260726"
RL_MODEL_PATH = FROZEN_DIR / "selected_model.zip"
RL_VECNORMALIZE_PATH = FROZEN_DIR / "selected_vecnormalize.pkl"

DEFAULT_SEED = 100_000  # matches training_extension/view_ppo_parameters.py's own default

PANEL_WIDTH = 480
PANEL_HEIGHT = 360
PANEL_GAP = 8
TOTAL_STEPS = 850  # scripts/view_baselines_LEVEL03_v031!.py's own fixed loop length
BASELINE_RELEASE_STEP = 406  # hardcoded in that script
SIM2REAL_STEPS_CAP = 1100  # BasketballResidualEnv.max_policy_steps; typical runs finish ~426
SIM2REAL_TAIL_STEPS = 40  # extra post-episode physics steps so the viewer sees the ball settle


def _load_baseline_module():
    """Dynamically imports scripts/view_baselines_LEVEL03_v031!.py, the
    same technique training_extension/derived_baseline.py already uses for
    this exact file (the "!" in the filename isn't a valid Python module
    name, so it can't be a normal import). Its own `if __name__ == "__main__"`
    guard means importing it doesn't launch the interactive viewer."""
    spec = importlib.util.spec_from_file_location("level03_baseline_script", BASELINE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load baseline script: {BASELINE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class BaselineShotMetrics:
    crossed_hoop: bool
    hoop_crossing_speed_mps: float | None
    max_impact_force_n: float
    max_pitch_deg: float
    max_roll_deg: float
    max_yaw_deg: float
    final_distance_m: float
    steps: int
    seed: int


@dataclass
class RLShotMetrics:
    success: bool
    crossing_error_cm: float | None
    touched_backboard: bool
    has_fallen: bool
    airborne_distance_m: float
    release_step: int | None
    steps: int
    reward_sum: float
    seed: int
    # Same 4 metrics BaselineShotMetrics reports, computed the same way,
    # now that training_extension/basketball_env.py's BasketballResidualEnv
    # exposes them too (feature/rl-on-lv3 commit "Expose Level 03
    # presentation metrics") -- field names match BaselineShotMetrics's so
    # the page can render both sides as one shared table.
    hoop_crossing_speed_mps: float | None
    max_impact_force_n: float
    max_pitch_deg: float
    max_roll_deg: float
    max_yaw_deg: float
    final_distance_m: float


@dataclass
class Level03ComparisonMetrics:
    baseline: BaselineShotMetrics
    rl: RLShotMetrics


@dataclass
class Level03ComparisonDone:
    metrics: Level03ComparisonMetrics


@dataclass
class Sim2RealComparisonMetrics:
    nominal: RLShotMetrics
    sim2real: RLShotMetrics


@dataclass
class Sim2RealComparisonDone:
    metrics: Sim2RealComparisonMetrics


Level03StreamEvent = Union[StreamFrame, Level03ComparisonDone, Sim2RealComparisonDone]


class BaselineSlot:
    """Runs scripts/view_baselines_LEVEL03_v031!.py's own simulation loop,
    verbatim, headlessly. Every physics/control line below is copied from
    that script's view_baseline() function -- only the interactive-viewer
    plumbing (mujoco.viewer.launch_passive, the outer "while
    viewer.is_running()") is replaced with offscreen render + explicit
    step-by-step control for streaming."""

    def __init__(self, renderer: mujoco.Renderer, camera: mujoco.MjvCamera):
        module = _load_baseline_module()
        self.env = module.G1FixedBodyThrowEnv(xml_path=str(BASELINE_SCENE_PATH))
        self.policy = module.OptionDBasketballPolicy(self.env)
        self._get_torso_tilt = module.get_torso_tilt
        self.renderer = renderer
        self.camera = camera

        self.target_body_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "throw_target")
        self.pelvis_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.ball_body_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "throw_ball")
        self.weld_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_EQUALITY, "hold_throw_ball")
        self.control_dt = getattr(self.env, "control_dt", 0.02)

        self.done = False
        self.max_pitch = self.max_roll = self.max_yaw = 0.0
        self.last_ball_pos = np.zeros(3)
        self.last_time: float | None = None
        self.ball_released = False
        self.ball_crossed_hoop = False
        self.hoop_crossing_speed = 0.0
        self.max_hoop_impact_force = 0.0

    def reset(self, seed: int) -> None:
        self.env.reset(seed=seed)
        self.policy.reset()
        # Same target placement/pose reset view_baseline() does before its loop.
        self.env.model.body_pos[self.target_body_id][0] = 1.8
        self.env.model.body_pos[self.target_body_id][2] = 1.2
        self.env.data.qpos[:3] = [0.0, 0.0, 0.81]
        self.env.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)

        self.done = False
        self.max_pitch = self.max_roll = self.max_yaw = 0.0
        self.last_ball_pos = self.env.data.xpos[self.ball_body_id].copy()
        self.last_time = None
        self.ball_released = False
        self.ball_crossed_hoop = False
        self.hoop_crossing_speed = 0.0
        self.max_hoop_impact_force = 0.0

    def step(self) -> None:
        if self.done:
            return  # the reference script does nothing past its own 850-step loop either
        self.policy.apply_controls()

        if self.policy.step_count == BASELINE_RELEASE_STEP and not self.ball_released:
            self.env.data.eq_active[self.weld_id] = 0
            self.ball_released = True

        pitch, roll, yaw = self._get_torso_tilt(self.env.model, self.env.data)
        torque_pitch = np.clip((0.0 - pitch) * 100.0 - self.env.data.qvel[4] * 20.0, -200.0, 200.0)
        torque_roll = np.clip((0.0 - roll) * 100.0 - self.env.data.qvel[3] * 20.0, -200.0, 200.0)
        torque_yaw = np.clip((0.0 - yaw) * 50.0 - self.env.data.qvel[5] * 10.0, -100.0, 100.0)
        force_y = np.clip((0.0 - self.env.data.qpos[1]) * 50.0 - self.env.data.qvel[1] * 10.0, -50.0, 50.0)
        self.env.data.xfrc_applied[self.pelvis_id, 3] = torque_roll
        self.env.data.xfrc_applied[self.pelvis_id, 4] = torque_pitch
        self.env.data.xfrc_applied[self.pelvis_id, 5] = torque_yaw
        self.env.data.xfrc_applied[self.pelvis_id, 1] = force_y

        mujoco.mj_step(self.env.model, self.env.data)

        current_time = self.env.data.time
        dt = current_time - self.last_time if self.last_time is not None else self.control_dt
        current_ball_pos = self.env.data.xpos[self.ball_body_id].copy()

        if self.ball_released:
            hx, hy, hz = self.env.data.xpos[self.target_body_id]
            if (
                self.last_ball_pos[2] > hz
                and current_ball_pos[2] <= hz
                and not self.ball_crossed_hoop
            ):
                dist_to_center = np.hypot(current_ball_pos[0] - hx, current_ball_pos[1] - hy)
                if dist_to_center < 0.15:
                    self.ball_crossed_hoop = True
                    if dt > 0:
                        ball_vel = (current_ball_pos - self.last_ball_pos) / dt
                        self.hoop_crossing_speed = float(np.linalg.norm(ball_vel))
            for i in range(self.env.data.ncon):
                contact = self.env.data.contact[i]
                g1 = mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                g2 = mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                if g1 and g2 and (("throw_ball" in g1 and "rim" in g2) or ("throw_ball" in g2 and "rim" in g1)):
                    c_array = np.zeros(6, dtype=np.float64)
                    mujoco.mj_contactForce(self.env.model, self.env.data, i, c_array)
                    self.max_hoop_impact_force = max(self.max_hoop_impact_force, abs(float(c_array[0])))

        self.last_ball_pos = current_ball_pos
        self.last_time = current_time
        self.max_pitch = max(self.max_pitch, abs(pitch))
        self.max_roll = max(self.max_roll, abs(roll))
        self.max_yaw = max(self.max_yaw, abs(yaw))

        if self.policy.step_count >= TOTAL_STEPS:
            self.done = True

    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.env.data, camera=self.camera)
        return self.renderer.render()

    def metrics(self, seed: int) -> BaselineShotMetrics:
        final_ball_pos = self.env.data.body("throw_ball").xpos
        final_target_pos = self.env.data.body("throw_target").xpos
        final_distance = float(np.linalg.norm(final_target_pos - final_ball_pos))
        return BaselineShotMetrics(
            crossed_hoop=self.ball_crossed_hoop,
            hoop_crossing_speed_mps=self.hoop_crossing_speed if self.ball_crossed_hoop else None,
            max_impact_force_n=self.max_hoop_impact_force,
            max_pitch_deg=self.max_pitch,
            max_roll_deg=self.max_roll,
            max_yaw_deg=self.max_yaw,
            final_distance_m=final_distance,
            steps=int(self.policy.step_count),
            seed=seed,
        )


class RLSlot:
    """Runs training_extension/view_ppo_parameters.py's own logic: predicts
    one parameter residual from the initial observation via the trained
    PPO model + VecNormalize, then plays out the whole walk-dip-throw
    sequence via optimize_direct.controller_action() using those fixed
    parameters -- identical to that script, just headless."""

    def __init__(self, renderer: mujoco.Renderer, camera: mujoco.MjvCamera):
        self.shot_env = SACShotParameterEnv()
        self.env = self.shot_env.base  # BasketballResidualEnv
        vector_env = DummyVecEnv([lambda: self.shot_env])
        vector_env = VecNormalize.load(str(RL_VECNORMALIZE_PATH), vector_env)
        vector_env.training = False
        vector_env.norm_reward = False
        self.vector_env = vector_env
        self.model = PPO.load(RL_MODEL_PATH, device="cpu")
        self.renderer = renderer
        self.camera = camera
        self.control_dt = self.env.control_substeps * self.env.model.opt.timestep

        self.parameters: np.ndarray | None = None
        self.done = False
        self.info: dict = {}
        self.reward_sum = 0.0

    def reset(self, seed: int) -> None:
        self.vector_env.seed(seed)
        observation = self.vector_env.reset()
        residual, _ = self.model.predict(observation, deterministic=True)
        self.parameters = self.shot_env.expert_parameters + self.shot_env.parameter_scales * residual[0]
        self.done = False
        self.info = {}
        self.reward_sum = 0.0

    def step(self) -> None:
        if self.done:
            # BasketballResidualEnv.step() applies _apply_peer_stabilizer()
            # before every mj_step(); skip it and the robot collapses once
            # the episode ends even though it never actually fell.
            for _ in range(self.env.control_substeps):
                self.env._apply_peer_stabilizer()
                mujoco.mj_step(self.env.model, self.env.data)
                self._track_post_episode_instrumentation()
            return
        action = controller_action(self.env, self.parameters)
        _, reward, terminated, truncated, self.info = self.env.step(action)
        self.reward_sum += float(reward)
        if terminated or truncated:
            self.done = True

    def _track_post_episode_instrumentation(self) -> None:
        """BasketballResidualEnv.step()'s own substep loop is what updates
        max_abs_torso_tilt_deg and max_rim_impact_force (see feature/rl-on-lv3's
        "Expose Level 03 presentation metrics" commit) -- calling raw
        mj_step() during the post-episode tail above bypasses that, so
        without this those two running-maxes would freeze at whatever they
        were at termination instead of covering the same full window the
        baseline script tracks them over (its entire 850-step loop,
        settling included)."""
        env = self.env
        env.max_abs_torso_tilt_deg = np.maximum(
            env.max_abs_torso_tilt_deg,
            np.abs(env._tilt_degrees(env.data.xquat[env.pelvis_id])),
        )
        for i in range(env.data.ncon):
            contact = env.data.contact[i]
            pair = {int(contact.geom1), int(contact.geom2)}
            if env.peer_env.ball_geom_id in pair and bool(pair & env.rim_geom_id_set):
                contact_force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(env.model, env.data, i, contact_force)
                env.max_rim_impact_force = max(env.max_rim_impact_force, abs(float(contact_force[0])))

    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.env.data, camera=self.camera)
        return self.renderer.render()

    def metrics(self, seed: int) -> RLShotMetrics:
        crossing_error = self.info.get("crossing_xy_error")
        env = self.env
        # Read live from env state, not the (possibly stale, pre-tail)
        # self.info snapshot -- matching how BaselineSlot.metrics() reads
        # its own final_distance_m fresh rather than from a stored value.
        final_distance = float(np.linalg.norm(env.data.xpos[env.ball_id] - env.target))
        return RLShotMetrics(
            success=bool(self.info.get("success", False)),
            crossing_error_cm=None if crossing_error is None else float(crossing_error) * 100.0,
            touched_backboard=bool(self.info.get("touched_backboard", False)),
            has_fallen=bool(self.info.get("has_fallen", False)),
            airborne_distance_m=float(self.info.get("airborne_horizontal_distance", 0.0)),
            release_step=self.info.get("release_step"),
            steps=int(self.env.policy.step_count),
            reward_sum=self.reward_sum,
            hoop_crossing_speed_mps=self.info.get("hoop_crossing_speed_m_s"),
            max_impact_force_n=float(env.max_rim_impact_force),
            max_pitch_deg=float(env.max_abs_torso_tilt_deg[0]),
            max_roll_deg=float(env.max_abs_torso_tilt_deg[1]),
            max_yaw_deg=float(env.max_abs_torso_tilt_deg[2]),
            final_distance_m=final_distance,
            seed=seed,
        )


class NoisyRLSlot(RLSlot):
    """Runs scripts/level_3_view_noisy.py's own Sim2Real domain-randomization
    gauntlet on top of the same trained RL policy: every reset, physics
    parameters (joint friction/damping, actuator force range, contact
    solref/solimp, floor friction) and the hoop target position are
    randomly perturbed, using that script's own perturbation ranges
    verbatim (0.7-1.3x friction/damping, 0.85-1.0x actuator force, 0.5-2.0x
    contact stiffness/impedance, 0.5-1.5x floor friction, +/-3cm target
    noise on x/y only) -- which happen to match Level 02's
    G1RobustnessEnv._enable_all_defaults() ranges.

    Matching that script exactly: the observation used for the model's
    one-shot residual prediction is captured from reset() *before* these
    perturbations are applied (perturbing physics/target doesn't retroactively
    change an already-computed observation), and the randomization draw
    itself uses bare np.random, not the seeded env RNG -- so, like Level 02's
    Sim2Real page, the nominal side's initial state is reproducible per seed
    but the randomization strength on this side varies every run by design.
    """

    def __init__(self, renderer: mujoco.Renderer, camera: mujoco.MjvCamera):
        super().__init__(renderer, camera)
        m = self.env.model
        self._baseline_frictionloss = m.dof_frictionloss.copy()
        self._baseline_damping = m.dof_damping.copy()
        self._baseline_forcerange = m.actuator_forcerange.copy()
        self._baseline_solref = m.opt.o_solref.copy()
        self._baseline_solimp = m.opt.o_solimp.copy()
        self._floor_geom_ids = [i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE]
        self._baseline_target = self.env.target.copy()

    def reset(self, seed: int) -> None:
        self.vector_env.seed(seed)
        observation = self.vector_env.reset()

        m = self.env.model
        m.dof_frictionloss[:] = self._baseline_frictionloss * np.random.uniform(0.7, 1.3)
        m.dof_damping[:] = self._baseline_damping * np.random.uniform(0.7, 1.3)
        m.actuator_forcerange[:] = self._baseline_forcerange * np.random.uniform(0.85, 1.0)
        m.opt.o_solref[0] = self._baseline_solref[0] * np.random.uniform(0.5, 2.0)
        m.opt.o_solimp[0] = self._baseline_solimp[0] * np.random.uniform(0.5, 2.0)
        for gid in self._floor_geom_ids:
            m.geom_friction[gid, 0] *= np.random.uniform(0.5, 1.5)
        self.env.target = self._baseline_target + np.random.normal(0, 0.03, 3)
        self.env.target[2] = self._baseline_target[2]
        mujoco.mj_forward(m, self.env.data)

        residual, _ = self.model.predict(observation, deterministic=True)
        self.parameters = self.shot_env.expert_parameters + self.shot_env.parameter_scales * residual[0]
        self.done = False
        self.info = {}
        self.reward_sum = 0.0


class Level03Runner:
    """Loads both scripts' envs + the RL model once and reuses them."""

    def __init__(self):
        baseline_camera = mujoco.MjvCamera()
        baseline_camera.lookat[:] = [0.9, 0.0, 1.0]
        baseline_camera.distance = 4.3
        baseline_camera.azimuth = 90
        baseline_camera.elevation = -8

        rl_camera = mujoco.MjvCamera()
        rl_camera.lookat[:] = [1.1, 0.0, 1.0]
        rl_camera.distance = 4.3
        rl_camera.azimuth = 90
        rl_camera.elevation = -8

        self.baseline = BaselineSlot(
            renderer=None,  # set below, after env/model exist
            camera=baseline_camera,
        )
        self.baseline.renderer = mujoco.Renderer(self.baseline.env.model, height=PANEL_HEIGHT, width=PANEL_WIDTH)

        self.rl = RLSlot(renderer=None, camera=rl_camera)
        self.rl.renderer = mujoco.Renderer(self.rl.env.model, height=PANEL_HEIGHT, width=PANEL_WIDTH)

        nominal_camera = mujoco.MjvCamera()
        nominal_camera.lookat[:] = [1.1, 0.0, 1.0]
        nominal_camera.distance = 4.3
        nominal_camera.azimuth = 90
        nominal_camera.elevation = -8

        sim2real_camera = mujoco.MjvCamera()
        sim2real_camera.lookat[:] = [1.1, 0.0, 1.0]
        sim2real_camera.distance = 4.3
        sim2real_camera.azimuth = 90
        sim2real_camera.elevation = -8

        self.nominal = RLSlot(renderer=None, camera=nominal_camera)
        self.nominal.renderer = mujoco.Renderer(self.nominal.env.model, height=PANEL_HEIGHT, width=PANEL_WIDTH)

        self.sim2real = NoisyRLSlot(renderer=None, camera=sim2real_camera)
        self.sim2real.renderer = mujoco.Renderer(self.sim2real.env.model, height=PANEL_HEIGHT, width=PANEL_WIDTH)

    def _stream_pair(
        self,
        left,
        right,
        seed: int,
        total_steps: int,
        tail_steps: int,
        break_when_both_done: bool,
    ) -> Iterator[StreamFrame]:
        left.reset(seed)
        right.reset(seed)

        control_dt = left.control_dt
        start_time = time.monotonic()
        frame_idx = 0

        def composite_frame() -> np.ndarray:
            nonlocal frame_idx
            due_at = start_time + frame_idx * control_dt
            remaining = due_at - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            frame_idx += 1
            gap = np.full((PANEL_HEIGHT, PANEL_GAP, 3), 20, dtype=np.uint8)
            return np.concatenate([left.render(), gap, right.render()], axis=1)

        yield StreamFrame(composite_frame())
        for _ in range(total_steps):
            left.step()
            right.step()
            yield StreamFrame(composite_frame())
            if break_when_both_done and left.done and right.done:
                break

        for _ in range(tail_steps):
            left.step()
            right.step()
            yield StreamFrame(composite_frame())

    def run_comparison_stream(self, seed: int | None = None) -> Iterator[Level03StreamEvent]:
        """Scripted baseline (left) vs RL policy (right). Runs the
        baseline's own fixed 850-step loop length exactly; no separate tail
        needed since both slots already self-manage staying alive/stable
        past their own termination (BaselineSlot no-ops past 850, RLSlot
        holds a stabilized idle pose)."""
        seed = DEFAULT_SEED if seed is None else seed
        yield from self._stream_pair(
            self.baseline, self.rl, seed, total_steps=TOTAL_STEPS, tail_steps=0, break_when_both_done=False
        )
        yield Level03ComparisonDone(
            Level03ComparisonMetrics(
                baseline=self.baseline.metrics(seed),
                rl=self.rl.metrics(seed),
            )
        )

    def run_sim2real_stream(self, seed: int | None = None) -> Iterator[Level03StreamEvent]:
        """The same trained RL policy in a clean/nominal env (left) vs
        scripts/level_3_view_noisy.py's Sim2Real domain-randomization
        gauntlet (right), same seed. Both sides are BasketballResidualEnv
        instances that terminate early (~426 steps typical) well before the
        1100-step hard cap, so this breaks out once both are done rather
        than running the full baseline-comparison's fixed 850."""
        seed = DEFAULT_SEED if seed is None else seed
        yield from self._stream_pair(
            self.nominal,
            self.sim2real,
            seed,
            total_steps=SIM2REAL_STEPS_CAP,
            tail_steps=SIM2REAL_TAIL_STEPS,
            break_when_both_done=True,
        )
        yield Sim2RealComparisonDone(
            Sim2RealComparisonMetrics(
                nominal=self.nominal.metrics(seed),
                sim2real=self.sim2real.metrics(seed),
            )
        )
