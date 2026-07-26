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


@dataclass
class Level03ComparisonMetrics:
    baseline: BaselineShotMetrics
    rl: RLShotMetrics


@dataclass
class Level03ComparisonDone:
    metrics: Level03ComparisonMetrics


Level03StreamEvent = Union[StreamFrame, Level03ComparisonDone]


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
            return
        action = controller_action(self.env, self.parameters)
        _, reward, terminated, truncated, self.info = self.env.step(action)
        self.reward_sum += float(reward)
        if terminated or truncated:
            self.done = True

    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.env.data, camera=self.camera)
        return self.renderer.render()

    def metrics(self, seed: int) -> RLShotMetrics:
        crossing_error = self.info.get("crossing_xy_error")
        return RLShotMetrics(
            success=bool(self.info.get("success", False)),
            crossing_error_cm=None if crossing_error is None else float(crossing_error) * 100.0,
            touched_backboard=bool(self.info.get("touched_backboard", False)),
            has_fallen=bool(self.info.get("has_fallen", False)),
            airborne_distance_m=float(self.info.get("airborne_horizontal_distance", 0.0)),
            release_step=self.info.get("release_step"),
            steps=int(self.env.policy.step_count),
            reward_sum=self.reward_sum,
            seed=seed,
        )


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

    def run_comparison_stream(self, seed: int | None = None) -> Iterator[Level03StreamEvent]:
        seed = DEFAULT_SEED if seed is None else seed
        self.baseline.reset(seed)
        self.rl.reset(seed)

        control_dt = self.baseline.control_dt
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
            return np.concatenate([self.baseline.render(), gap, self.rl.render()], axis=1)

        yield StreamFrame(composite_frame())
        for _ in range(TOTAL_STEPS):
            self.baseline.step()
            self.rl.step()
            yield StreamFrame(composite_frame())

        yield Level03ComparisonDone(
            Level03ComparisonMetrics(
                baseline=self.baseline.metrics(seed),
                rl=self.rl.metrics(seed),
            )
        )
