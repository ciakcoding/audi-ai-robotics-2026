"""Runs one Task 2 PPO episode headlessly and renders it to an MP4.

Reuses envs/ppo_throw_env.py and outputs/models/selected/best/best_model.zip
exactly as evaluation/evaluate_ppo.py does -- same env class, same
residual_scale, same deterministic policy.predict() loop -- so the numbers
shown in the UI match the numbers in the reports.

The only deviation from the eval scripts is the mesh source: the top-level
assets/scene_throw.xml that PPOThrowEnv normally loads currently fails to
resolve its STL meshes on this checkout (a pre-existing meshdir bug in
assets/g1.xml, unrelated to this webapp). assets/unitree_g1/scene_throw.xml
is a self-contained, working copy of the same robot/scene, so WebPPOThrowEnv
points there instead. It also adds a tiny air density/viscosity <option>
the top-level scene doesn't have, which can nudge the ball's flight
slightly -- worth knowing if displayed numbers drift from the frozen
evaluation report.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Union

import imageio.v2 as imageio
import mujoco
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from envs.g1_fixed_body_throw_env import G1FixedBodyThrowEnv
from envs.g1_robustness_env import G1RobustnessEnv
from envs.ppo_throw_env import PPOThrowEnv

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs" / "models" / "selected" / "best" / "best_model.zip"
VIDEO_DIR = Path(__file__).resolve().parent / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

RESIDUAL_SCALE = 0.2
DEFAULT_SEED = 2026

RENDER_WIDTH = 640
RENDER_HEIGHT = 480
TAIL_STEPS = 60  # extra post-episode physics steps so the viewer sees the bounce

PANEL_WIDTH = 480
PANEL_HEIGHT = 360
PANEL_GAP = 8


class WebPPOThrowEnv(PPOThrowEnv):
    """PPOThrowEnv on the self-contained assets/unitree_g1 mesh copy.

    Only __init__ differs from PPOThrowEnv -- reset()/step() are inherited
    unchanged, so the residual-control and reward logic used for training
    and evaluation is untouched.
    """

    def __init__(self, residual_scale: float = RESIDUAL_SCALE):
        G1FixedBodyThrowEnv.__init__(
            self,
            xml_path=ROOT / "assets" / "unitree_g1" / "scene_throw.xml",
            learned_release=False,
        )
        self.residual_scale = float(residual_scale)
        self.extra_initial_joint_noise = 0.0
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.baseline_start = np.array(
            [0.9318, -0.7911, 0.0491, -0.1425, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self.baseline_end = np.array(
            [-1.0, 0.0964, 0.0072, -1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )


class WebRobustnessEnv(G1RobustnessEnv):
    """G1RobustnessEnv (Sim2Real domain randomization) on the same working
    assets/unitree_g1 mesh copy WebPPOThrowEnv uses.

    G1RobustnessEnv.__init__ calls PPOThrowEnv.__init__(residual_scale=...,
    extra_initial_joint_noise=...), which hardcodes the broken top-level
    scene path -- the same problem WebPPOThrowEnv works around above. There
    is no way to override just that one call cooperatively (G1RobustnessEnv
    always inherits directly from PPOThrowEnv, so Python's MRO can't be
    reshaped to insert a substitute in between), so this constructor
    replicates G1RobustnessEnv.__init__ verbatim, swapping only that one
    super().__init__() call for the same G1FixedBodyThrowEnv-direct
    construction WebPPOThrowEnv uses. Everything else below is copied
    unchanged from envs/g1_robustness_env.py -- keep the two in sync if
    that file's __init__ ever changes.
    """

    def __init__(
        self,
        residual_scale: float = RESIDUAL_SCALE,
        extra_initial_joint_noise: float = 0.0,
        enable_all: bool = False,
    ):
        # --- swapped-in construction (mirrors WebPPOThrowEnv.__init__) ---
        G1FixedBodyThrowEnv.__init__(
            self,
            xml_path=ROOT / "assets" / "unitree_g1" / "scene_throw.xml",
            learned_release=False,
        )
        self.residual_scale = float(residual_scale)
        self.extra_initial_joint_noise = float(extra_initial_joint_noise)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.baseline_start = np.array(
            [0.9318, -0.7911, 0.0491, -0.1425, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self.baseline_end = np.array(
            [-1.0, 0.0964, 0.0072, -1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        # --- end swap; rest copied verbatim from G1RobustnessEnv.__init__ ---

        self._floor_geom_ids = [
            i for i in range(self.model.ngeom)
            if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
        ]

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

        self.obs_noise = 0.0
        self.ball_mass_range = None
        self.joint_friction_range = None
        self.joint_damping_range = None
        self.floor_friction_range = None
        self.actuator_gain_range = None
        self.target_pos_noise = 0.0
        self.ball_size_range = None
        self.control_latency_steps = 0
        self.action_noise = 0.0
        self.push_probability = 0.0
        self.push_force_range = (-3.0, 3.0)
        self.contact_solref_range = None
        self.contact_solimp_range = None

        self._action_buffer = []
        self.current_randomization = {}

        if enable_all:
            self._enable_all_defaults()


@dataclass
class EpisodeResult:
    video_url: str
    landing_error_cm: float | None
    success: bool
    has_fallen: bool
    reward_sum: float
    steps: int
    release_time_s: float | None


@dataclass
class EpisodeMetrics:
    landing_error_cm: float | None
    success: bool
    has_fallen: bool
    reward_sum: float
    steps: int
    release_time_s: float | None
    seed: int


@dataclass
class ComparisonMetrics:
    baseline: EpisodeMetrics
    rl: EpisodeMetrics


@dataclass
class Sim2RealMetrics:
    nominal: EpisodeMetrics
    sim2real: EpisodeMetrics


@dataclass
class StreamFrame:
    image: np.ndarray


@dataclass
class StreamDone:
    metrics: EpisodeMetrics


@dataclass
class ComparisonDone:
    metrics: ComparisonMetrics


@dataclass
class Sim2RealDone:
    metrics: Sim2RealMetrics


StreamEvent = Union[StreamFrame, StreamDone, ComparisonDone, Sim2RealDone]


def _metrics_from_info(info: dict, reward_sum: float, steps: int, seed: int) -> EpisodeMetrics:
    landing_error = info.get("landing_error_xy")
    return EpisodeMetrics(
        landing_error_cm=None if landing_error is None else float(landing_error) * 100.0,
        success=bool(info.get("success", False)),
        has_fallen=bool(info.get("has_fallen", False)),
        reward_sum=reward_sum,
        steps=steps,
        release_time_s=info.get("release_time"),
        seed=seed,
    )


class _CompareSlot:
    """One side (baseline or RL) of the side-by-side comparison.

    policy=None means "no residual" -- PPOThrowEnv.step(zeros) applies the
    scripted baseline swing exactly (residual_scale * 0 == 0), which is how
    evaluation/compare_baseline_ppo.py represents the baseline too.
    """

    def __init__(self, env: "WebPPOThrowEnv", renderer: mujoco.Renderer, camera: mujoco.MjvCamera, policy):
        self.env = env
        self.renderer = renderer
        self.camera = camera
        self.policy = policy
        self.obs = None
        self.done = False
        self.info: dict = {}
        self.reward_sum = 0.0

    def reset(self, seed: int) -> None:
        self.obs, _ = self.env.reset(seed=seed)
        self.done = False
        self.info = {}
        self.reward_sum = 0.0

    def step(self) -> None:
        if self.done:
            for _ in range(self.env.frame_skip):
                mujoco.mj_step(self.env.model, self.env.data)
            return
        if self.policy is None:
            action = np.zeros(7, dtype=np.float32)
        else:
            action, _ = self.policy.predict(self.obs, deterministic=True)
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.reward_sum += float(reward)
        if terminated or truncated:
            self.done = True

    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.env.data, camera=self.camera)
        return self.renderer.render()

    def metrics(self, seed: int) -> EpisodeMetrics:
        return _metrics_from_info(self.info, self.reward_sum, self.env.step_count, seed)


class SimulationRunner:
    """Loads the env + policy once and reuses them across requests."""

    def __init__(self):
        self.env = WebPPOThrowEnv(residual_scale=RESIDUAL_SCALE)
        self.model = PPO.load(MODEL_PATH, device="cpu")
        self.renderer = mujoco.Renderer(self.env.model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
        self.camera = self._build_camera()

        panel_camera = self._build_camera()
        baseline_env = WebPPOThrowEnv(residual_scale=RESIDUAL_SCALE)
        rl_env = WebPPOThrowEnv(residual_scale=RESIDUAL_SCALE)
        self.baseline_slot = _CompareSlot(
            env=baseline_env,
            renderer=mujoco.Renderer(baseline_env.model, height=PANEL_HEIGHT, width=PANEL_WIDTH),
            camera=panel_camera,
            policy=None,
        )
        self.rl_slot = _CompareSlot(
            env=rl_env,
            renderer=mujoco.Renderer(rl_env.model, height=PANEL_HEIGHT, width=PANEL_WIDTH),
            camera=panel_camera,
            policy=self.model,
        )

        # Sim2Real pair: same trained RL policy on both sides, the only
        # difference is environment perturbation -- mirrors the "clean vs
        # noisy" comparison in scripts/evaluate_robustness.py.
        nominal_env = WebRobustnessEnv(residual_scale=RESIDUAL_SCALE, enable_all=False)
        sim2real_env = WebRobustnessEnv(residual_scale=RESIDUAL_SCALE, enable_all=True)
        self.nominal_slot = _CompareSlot(
            env=nominal_env,
            renderer=mujoco.Renderer(nominal_env.model, height=PANEL_HEIGHT, width=PANEL_WIDTH),
            camera=panel_camera,
            policy=self.model,
        )
        self.sim2real_slot = _CompareSlot(
            env=sim2real_env,
            renderer=mujoco.Renderer(sim2real_env.model, height=PANEL_HEIGHT, width=PANEL_WIDTH),
            camera=panel_camera,
            policy=self.model,
        )

    def _build_camera(self) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.5, 0.0, 0.85]
        cam.distance = 2.3
        cam.azimuth = 272
        cam.elevation = -12
        return cam

    def _frame(self) -> np.ndarray:
        self.renderer.update_scene(self.env.data, camera=self.camera)
        return self.renderer.render()

    def run_episode(self, seed: int | None = None) -> EpisodeResult:
        seed = DEFAULT_SEED if seed is None else seed
        env = self.env
        obs, _ = env.reset(seed=seed)
        frames = [self._frame()]
        terminated = truncated = False
        info: dict = {}
        reward_sum = 0.0
        while not (terminated or truncated):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            frames.append(self._frame())

        for _ in range(TAIL_STEPS):
            for _ in range(env.frame_skip):
                mujoco.mj_step(env.model, env.data)
            frames.append(self._frame())

        video_name = f"episode_{uuid.uuid4().hex}.mp4"
        video_path = VIDEO_DIR / video_name
        fps = round(1.0 / env.control_dt)
        imageio.mimwrite(video_path, frames, fps=fps, codec="libx264", quality=7)

        m = _metrics_from_info(info, reward_sum, env.step_count, seed)
        return EpisodeResult(
            video_url=f"/videos/{video_name}",
            landing_error_cm=m.landing_error_cm,
            success=m.success,
            has_fallen=m.has_fallen,
            reward_sum=m.reward_sum,
            steps=m.steps,
            release_time_s=m.release_time_s,
        )

    def run_episode_stream(self, seed: int | None = None) -> Iterator[StreamEvent]:
        """Like run_episode, but yields each frame as it's rendered (paced to
        real time via env.control_dt) instead of assembling an MP4. Used by
        the /ws/run websocket for live playback."""
        seed = DEFAULT_SEED if seed is None else seed
        env = self.env
        obs, _ = env.reset(seed=seed)
        terminated = truncated = False
        info: dict = {}
        reward_sum = 0.0
        start_time = time.monotonic()
        frame_idx = 0

        def next_frame() -> StreamFrame:
            nonlocal frame_idx
            due_at = start_time + frame_idx * env.control_dt
            remaining = due_at - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            frame_idx += 1
            return StreamFrame(self._frame())

        yield next_frame()
        while not (terminated or truncated):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            yield next_frame()

        for _ in range(TAIL_STEPS):
            for _ in range(env.frame_skip):
                mujoco.mj_step(env.model, env.data)
            yield next_frame()

        metrics = _metrics_from_info(info, reward_sum, env.step_count, seed)
        yield StreamDone(metrics)

    def _stream_pair(self, left: _CompareSlot, right: _CompareSlot, seed: int) -> Iterator[StreamFrame]:
        """Steps two slots through the same seed in lockstep, yielding one
        composite frame (left | right) per tick, paced to real time. Once a
        side finishes it keeps stepping physics-only (no policy/info
        updates) so the ball can be seen landing/rolling while the other
        side catches up. Does not yield a Done event -- callers build and
        yield their own, typed metrics event after exhausting this."""
        left.reset(seed)
        right.reset(seed)

        control_dt = left.env.control_dt
        main_steps = int(round(left.env.episode_time / control_dt))
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
        for _ in range(main_steps):
            left.step()
            right.step()
            yield StreamFrame(composite_frame())
            if left.done and right.done:
                break

        for _ in range(TAIL_STEPS):
            left.step()
            right.step()
            yield StreamFrame(composite_frame())

    def run_comparison_stream(self, seed: int | None = None) -> Iterator[StreamEvent]:
        """Scripted baseline (left) vs RL policy (right), same seed."""
        seed = DEFAULT_SEED if seed is None else seed
        yield from self._stream_pair(self.baseline_slot, self.rl_slot, seed)
        yield ComparisonDone(
            ComparisonMetrics(
                baseline=self.baseline_slot.metrics(seed),
                rl=self.rl_slot.metrics(seed),
            )
        )

    def run_sim2real_stream(self, seed: int | None = None) -> Iterator[StreamEvent]:
        """The same trained RL policy in a clean/nominal env (left) vs the
        full 7-parameter Sim2Real domain-randomization gauntlet (right),
        same seed -- a live version of scripts/evaluate_robustness.py's
        clean-vs-noisy comparison. The randomization draw itself (ball
        mass, friction, target offset, etc.) uses G1RobustnessEnv's own
        np.random calls, not the seeded env RNG, so it varies run to run
        even for a repeated seed -- matching the original evaluation
        script's behavior."""
        seed = DEFAULT_SEED if seed is None else seed
        yield from self._stream_pair(self.nominal_slot, self.sim2real_slot, seed)
        yield Sim2RealDone(
            Sim2RealMetrics(
                nominal=self.nominal_slot.metrics(seed),
                sim2real=self.sim2real_slot.metrics(seed),
            )
        )
