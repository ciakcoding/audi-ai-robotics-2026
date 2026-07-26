"""Minimal FastAPI GUI for the Task 2 G1 ball-throw policy.

Run locally with:
    MUJOCO_GL=glfw uvicorn webapp.app:app --reload
then open http://localhost:8000
"""

from __future__ import annotations

import asyncio
import io
import random
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.websockets import WebSocketState

from webapp.runner import ComparisonDone, Sim2RealDone, SimulationRunner, StreamDone, StreamFrame

APP_DIR = Path(__file__).resolve().parent
JPEG_QUALITY = 80

app = FastAPI(title="G1 Ball-Throw Demo")

_runner: SimulationRunner | None = None
_lock = threading.Lock()


def get_runner() -> SimulationRunner:
    global _runner
    if _runner is None:
        _runner = SimulationRunner()
    return _runner


def _encode_jpeg(frame) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


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


def _metrics_json(m) -> dict:
    return {
        "landing_error_cm": m.landing_error_cm,
        "success": m.success,
        "has_fallen": m.has_fallen,
        "reward_sum": m.reward_sum,
        "steps": m.steps,
        "release_time_s": m.release_time_s,
        "seed": m.seed,
    }


async def _stream_events(websocket: WebSocket, make_event_iter) -> None:
    """Runs make_event_iter() (a StreamEvent generator) on a worker thread
    under the shared-env lock, forwarding frames as binary JPEGs and the
    final metrics as one JSON message."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=4)

    def blocking_run():
        with _lock:
            for event in make_event_iter():
                asyncio.run_coroutine_threadsafe(queue.put(event), loop).result()
        asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    worker = loop.run_in_executor(None, blocking_run)
    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            if isinstance(event, StreamFrame):
                await websocket.send_bytes(_encode_jpeg(event.image))
            elif isinstance(event, StreamDone):
                await websocket.send_json({"type": "done", **_metrics_json(event.metrics)})
            elif isinstance(event, ComparisonDone):
                await websocket.send_json(
                    {
                        "type": "done",
                        "baseline": _metrics_json(event.metrics.baseline),
                        "rl": _metrics_json(event.metrics.rl),
                    }
                )
            elif isinstance(event, Sim2RealDone):
                await websocket.send_json(
                    {
                        "type": "done",
                        "nominal": _metrics_json(event.metrics.nominal),
                        "sim2real": _metrics_json(event.metrics.sim2real),
                    }
                )
    except WebSocketDisconnect:
        pass
    finally:
        await worker
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()


@app.websocket("/ws/run")
async def run_simulation_live(websocket: WebSocket):
    """Streams a single RL-policy episode frame-by-frame as JPEGs, then a
    final JSON metrics message, so the browser can play it back live on a
    <canvas> instead of waiting for a finished video."""
    await websocket.accept()
    seed = random.randint(0, 1_000_000)
    await _stream_events(websocket, lambda: get_runner().run_episode_stream(seed=seed))


@app.websocket("/ws/compare")
async def run_comparison_live(websocket: WebSocket):
    """Streams the scripted baseline and the RL policy side by side (one
    composite frame per tick, baseline left / RL right) on the same seed,
    then a final JSON message with both sides' metrics."""
    await websocket.accept()
    seed = random.randint(0, 1_000_000)
    await _stream_events(websocket, lambda: get_runner().run_comparison_stream(seed=seed))


@app.websocket("/ws/sim2real")
async def run_sim2real_live(websocket: WebSocket):
    """Streams the same RL policy in a clean/nominal env (left) vs the full
    Sim2Real domain-randomization gauntlet (right) on the same seed, then a
    final JSON message with both sides' metrics."""
    await websocket.accept()
    seed = random.randint(0, 1_000_000)
    await _stream_events(websocket, lambda: get_runner().run_sim2real_stream(seed=seed))


app.mount("/videos", StaticFiles(directory=APP_DIR / "videos"), name="videos")
app.mount("/", StaticFiles(directory=APP_DIR / "static", html=True), name="static")
