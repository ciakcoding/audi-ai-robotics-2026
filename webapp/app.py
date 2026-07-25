"""Minimal FastAPI GUI for the Task 2 G1 ball-throw policy.

Run locally with:
    MUJOCO_GL=glfw uvicorn webapp.app:app --reload
then open http://localhost:8000
"""

from __future__ import annotations

import random
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from webapp.runner import SimulationRunner

APP_DIR = Path(__file__).resolve().parent

app = FastAPI(title="G1 Ball-Throw Demo")

_runner: SimulationRunner | None = None
_lock = threading.Lock()


def get_runner() -> SimulationRunner:
    global _runner
    if _runner is None:
        _runner = SimulationRunner()
    return _runner


@app.post("/run")
def run_simulation():
    seed = random.randint(0, 1_000_000)
    with _lock:
        result = get_runner().run_episode(seed=seed)
    return {
        "video_url": result.video_url,
        "landing_error_cm": result.landing_error_cm,
        "success": result.success,
        "has_fallen": result.has_fallen,
        "reward_sum": result.reward_sum,
        "steps": result.steps,
        "release_time_s": result.release_time_s,
        "seed": seed,
    }


app.mount("/videos", StaticFiles(directory=APP_DIR / "videos"), name="videos")
app.mount("/", StaticFiles(directory=APP_DIR / "static", html=True), name="static")
