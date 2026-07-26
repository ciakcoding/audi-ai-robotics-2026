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
from webapp.runner_level03 import Level03ComparisonDone, Level03Runner, Sim2RealComparisonDone

APP_DIR = Path(__file__).resolve().parent
JPEG_QUALITY = 80

app = FastAPI(title="G1 Ball-Throw Demo")

_runner: SimulationRunner | None = None
_lock = threading.Lock()

_runner_l03: Level03Runner | None = None
_lock_l03 = threading.Lock()


def get_runner() -> SimulationRunner:
    global _runner
    if _runner is None:
        _runner = SimulationRunner()
    return _runner


def get_runner_l03() -> Level03Runner:
    global _runner_l03
    if _runner_l03 is None:
        _runner_l03 = Level03Runner()
    return _runner_l03


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


def _baseline_shot_json(m) -> dict:
    return {
        "crossed_hoop": m.crossed_hoop,
        "hoop_crossing_speed_mps": m.hoop_crossing_speed_mps,
        "max_impact_force_n": m.max_impact_force_n,
        "max_pitch_deg": m.max_pitch_deg,
        "max_roll_deg": m.max_roll_deg,
        "max_yaw_deg": m.max_yaw_deg,
        "final_distance_m": m.final_distance_m,
        "steps": m.steps,
        "seed": m.seed,
    }


def _rl_shot_json(m) -> dict:
    return {
        "success": m.success,
        "crossing_error_cm": m.crossing_error_cm,
        "touched_backboard": m.touched_backboard,
        "has_fallen": m.has_fallen,
        "airborne_distance_m": m.airborne_distance_m,
        "release_step": m.release_step,
        "steps": m.steps,
        "reward_sum": m.reward_sum,
        "seed": m.seed,
    }


async def _stream_events(websocket: WebSocket, make_event_iter, lock: threading.Lock = _lock) -> None:
    """Runs make_event_iter() (a StreamEvent generator) on a worker thread
    under the given lock, forwarding frames as binary JPEGs and the final
    metrics as one JSON message."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=4)

    def blocking_run():
        with lock:
            for event in make_event_iter():
                asyncio.run_coroutine_threadsafe(queue.put(event), loop).result()
        asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    worker = loop.run_in_executor(None, blocking_run)
    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            try:
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
                elif isinstance(event, Level03ComparisonDone):
                    await websocket.send_json(
                        {
                            "type": "done",
                            "baseline": _baseline_shot_json(event.metrics.baseline),
                            "rl": _rl_shot_json(event.metrics.rl),
                        }
                    )
                elif isinstance(event, Sim2RealComparisonDone):
                    await websocket.send_json(
                        {
                            "type": "done",
                            "nominal": _rl_shot_json(event.metrics.nominal),
                            "sim2real": _rl_shot_json(event.metrics.sim2real),
                        }
                    )
            except (WebSocketDisconnect, RuntimeError):
                # A half-closed transport surfaces as RuntimeError from
                # uvloop/uvicorn on send, not always as WebSocketDisconnect.
                # Either way the client is gone -- stop sending, but let the
                # producer finish naturally via the drain loop below rather
                # than letting this escape and take down the whole ASGI
                # worker.
                break
    except WebSocketDisconnect:
        pass
    finally:
        # If the client disconnected mid-stream, nothing above is calling
        # queue.get() anymore. blocking_run (still running, under `lock`)
        # would then block forever on queue.put() once the bounded queue
        # fills up, hanging this request AND holding the shared env lock
        # forever. Keep draining (discarding) until the worker thread
        # actually finishes so it can exit and release the lock even after
        # a client disconnect.
        while not worker.done():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.01)
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


@app.websocket("/ws/level03/compare")
async def run_level03_comparison_live(websocket: WebSocket):
    """Level 03: streams scripts/view_baselines_LEVEL03_v031!.py's own
    scripted basketball baseline (left) and training_extension's trained
    RL policy (right) -- each in its own script's own scene, not a shared
    env -- on the same seed, then a final JSON message with each side's
    own native metrics (the two scripts don't share a scoring contract).
    Fixed 850-step run (the baseline script's own loop length), so this
    takes ~25-30s wall-clock per click."""
    await websocket.accept()
    seed = random.randint(0, 1_000_000)
    await _stream_events(
        websocket,
        lambda: get_runner_l03().run_comparison_stream(seed=seed),
        lock=_lock_l03,
    )


@app.websocket("/ws/level03/sim2real")
async def run_level03_sim2real_live(websocket: WebSocket):
    """Level 03: streams the same trained RL policy in a clean/nominal env
    (left) vs scripts/level_3_view_noisy.py's Sim2Real domain-randomization
    gauntlet (right) on the same seed, then a final JSON message with both
    sides' metrics (same schema this time -- both sides are
    BasketballResidualEnv)."""
    await websocket.accept()
    seed = random.randint(0, 1_000_000)
    await _stream_events(
        websocket,
        lambda: get_runner_l03().run_sim2real_stream(seed=seed),
        lock=_lock_l03,
    )


app.mount("/videos", StaticFiles(directory=APP_DIR / "videos"), name="videos")
app.mount("/", StaticFiles(directory=APP_DIR / "static", html=True), name="static")
