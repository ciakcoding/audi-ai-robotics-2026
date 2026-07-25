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


class SimulationRunner:
    """Loads the env + policy once and reuses them across requests."""

    def __init__(self):
        self.env = WebPPOThrowEnv(residual_scale=RESIDUAL_SCALE)
        self.model = PPO.load(MODEL_PATH, device="cpu")
        self.renderer = mujoco.Renderer(self.env.model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
        self.camera = self._build_camera()

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

        landing_error = info.get("landing_error_xy")
        return EpisodeResult(
            video_url=f"/videos/{video_name}",
            landing_error_cm=None if landing_error is None else float(landing_error) * 100.0,
            success=bool(info.get("success", False)),
            has_fallen=bool(info.get("has_fallen", False)),
            reward_sum=reward_sum,
            steps=env.step_count,
            release_time_s=info.get("release_time"),
        )
