# G1 Ball-Throw Demo — Web GUI

A small local website that lets you watch the trained Task 2 policy throw
the ball, live, without touching a terminal. Click **Run Simulation** and
the browser shows the Unitree G1 arm swinging and releasing the ball in
real time, followed by the episode's metrics.

## What's actually happening

1. **You click "Run Simulation".** The browser opens a WebSocket connection
   to the FastAPI backend at `/ws/run`.
2. **The backend runs one fresh episode of the real simulation.** It loads
   the exact same environment and policy the project's evaluation scripts
   use:
   - Environment: `envs/ppo_throw_env.py` (`PPOThrowEnv`) — the residual
     control wrapper around the frozen Task 1 baseline.
   - Policy: `outputs/models/selected/best/best_model.zip`, loaded with
     Stable-Baselines3's `PPO.load(...)`.
   - Loop: `env.reset(seed=<random>)`, then repeated
     `model.predict(obs, deterministic=True)` → `env.step(action)` until the
     episode ends — identical to `evaluation/evaluate_ppo.py`.
   - A new random seed is picked per click, so every run is a genuinely new
     episode (slightly different initial arm noise), not a replay.
3. **Each simulation step is rendered and streamed immediately.** After
   every `env.step()`, MuJoCo's offscreen renderer (`mujoco.Renderer`)
   captures a frame, it's JPEG-encoded, and sent to the browser over the
   WebSocket as soon as it's ready — paced to real time (`env.control_dt`,
   20 ms/step) so playback speed matches the physics, not sped up or
   choppy.
4. **The browser draws frames onto a `<canvas>`** as they arrive — this is
   why it feels "live" rather than "wait, then play a video." There is no
   pre-rendered file involved in this path.
5. **After release, a short extra tail** (60 physics steps, no policy
   action) keeps streaming so you can watch the ball actually land/bounce,
   even though the episode technically terminates at first contact.
6. **When the episode ends**, the backend sends one final JSON message with
   the real metrics pulled from the same `info` dict the eval scripts use:
   landing error (cm), success, whether the robot fell, summed reward,
   step count, release time, and the seed used.

There's also a `POST /run` REST endpoint that does the same simulation but
writes an MP4 to `webapp/videos/` and returns its URL instead of streaming —
kept around as a fallback/download-friendly option; the page itself uses the
live WebSocket path by default.

## Why it points at `assets/unitree_g1/scene_throw.xml`

The environment class normally loads `assets/scene_throw.xml`, but that
scene's included `assets/g1.xml` currently has a `meshdir` that resolves to
a nonexistent `assets/assets/` folder on this checkout (pre-existing bug,
unrelated to this webapp). `assets/unitree_g1/scene_throw.xml` is a working,
self-contained copy of the same robot and scene, so `WebPPOThrowEnv` in
`webapp/runner.py` loads that instead — it only overrides `__init__`;
`reset()`/`step()` are inherited unchanged from `PPOThrowEnv`, so the
control/reward logic is untouched. That scene does add a small air
density/viscosity `<option>` the top-level one lacks, which can nudge
numbers slightly (observed ~0.68–0.83 cm landing error vs. the frozen
report's 0.674 cm mean) — worth knowing if you're comparing against the
official evaluation report.

## Files

```text
webapp/
  app.py          FastAPI app: POST /run, WS /ws/run, static file mounts
  runner.py       SimulationRunner: loads env + policy once, runs episodes
  static/
    index.html    The GUI (Audi red/black themed) — canvas + metrics panel
  videos/         Generated MP4s from POST /run (gitignored)
```

## Run it

```bash
source .venv/bin/activate
uvicorn webapp.app:app --reload
```

Open `http://localhost:8000` and click **Run Simulation**.

`MUJOCO_GL` controls the rendering backend and needs no code changes:
- Local machine with a display (default here): unset, or `glfw`.
- Headless server, CPU only: `MUJOCO_GL=osmesa`.
- Headless server with GPU: `MUJOCO_GL=egl`.

## Known limitations (v1)

- Single shared MuJoCo environment/renderer instance — concurrent runs from
  multiple browser tabs are serialized behind a lock, not parallelized.
- No deployment/containerization yet (`Dockerfile` is a planned next step).
