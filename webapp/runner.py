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
class StreamFrame:
    image: np.ndarray


@dataclass
class StreamDone:
    metrics: EpisodeMetrics


@dataclass
class ComparisonDone:
    metrics: ComparisonMetrics


StreamEvent = Union[StreamFrame, StreamDone, ComparisonDone]


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

    def _build_camera(self) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.35, 0.0, 0.9]
        cam.distance = 2.6
        cam.azimuth = 140
        cam.elevation = -20
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

    def run_comparison_stream(self, seed: int | None = None) -> Iterator[StreamEvent]:
        """Steps the scripted baseline and the RL policy through the same
        seed in lockstep, one composite frame (baseline left, RL right) per
        tick, paced to real time. Once a side finishes it keeps stepping
        physics-only (no policy/info updates) so the ball can be seen
        landing/rolling while the other side catches up."""
        seed = DEFAULT_SEED if seed is None else seed
        self.baseline_slot.reset(seed)
        self.rl_slot.reset(seed)

        control_dt = self.baseline_slot.env.control_dt
        main_steps = int(round(self.baseline_slot.env.episode_time / control_dt))
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
            left = self.baseline_slot.render()
            right = self.rl_slot.render()
            return np.concatenate([left, gap, right], axis=1)

        yield StreamFrame(composite_frame())
        for _ in range(main_steps):
            self.baseline_slot.step()
            self.rl_slot.step()
            yield StreamFrame(composite_frame())
            if self.baseline_slot.done and self.rl_slot.done:
                break

        for _ in range(TAIL_STEPS):
            self.baseline_slot.step()
            self.rl_slot.step()
            yield StreamFrame(composite_frame())

        yield ComparisonDone(
            ComparisonMetrics(
                baseline=self.baseline_slot.metrics(seed),
                rl=self.rl_slot.metrics(seed),
            )
        )
