# G1 Ball-Throw Demo — Web GUI (Level 02)

**This entire app is Level 02 only**: throw the ball to a fixed target
while the G1 balances (pelvis stays up, no locomotion). It is *not* Level 03
(walk forward while throwing) — that task was descoped from this repo
(see the "Remove Level 3 walk+throw experiments" commit) but is expected to
come back later. When it does, it will need its own env, its own trained
policy, and almost certainly its own page/route here — the code in this
directory (`WebPPOThrowEnv`, the baseline-vs-RL comparison, the metrics
shown) is written specifically around the Level 02 task and does not
generalize to walking. Don't try to repoint it at a Level 03 env without
rethinking the comparison logic and metrics from scratch.

A small local website that runs the scripted Task 1 baseline and the
trained Task 2 RL policy side by side, live, without touching a terminal.
Click **Run Simulation** and the browser shows both controllers throwing
the ball from the same starting conditions in real time — baseline on the
left, RL policy on the right — followed by a metrics comparison table.

## What's actually happening

1. **You click "Run Simulation".** The browser opens a WebSocket connection
   to the FastAPI backend at `/ws/compare`.
2. **The backend runs two fresh episodes in lockstep, on the same seed.**
   Both sides use `envs/ppo_throw_env.py` (`PPOThrowEnv`), the residual
   control wrapper around the frozen Task 1 baseline:
   - **Baseline side:** stepped with a zero residual action every tick.
     Since `PPOThrowEnv.step()` applies
     `baseline_swing + residual_scale * residual`, feeding zeros reproduces
     the scripted swing exactly — the same representation
     `evaluation/compare_baseline_ppo.py` uses for "baseline".
   - **RL side:** stepped with `outputs/models/selected/best/best_model.zip`
     (loaded via Stable-Baselines3's `PPO.load(...)`),
     `model.predict(obs, deterministic=True)` each tick — identical to
     `evaluation/evaluate_ppo.py`.
   - Both envs call `reset(seed=<same random seed>)`, so any difference in
     outcome comes from the policy, not from different initial noise. A new
     random seed is picked per click, so every run is a genuinely new pair
     of episodes, not a replay.
3. **Each tick, both sides are rendered and composited into one frame.**
   MuJoCo's offscreen renderer (`mujoco.Renderer`) captures a frame from
   each side's own `MjData`, they're concatenated left/right into a single
   image, JPEG-encoded, and sent to the browser over the WebSocket — paced
   to real time (`env.control_dt`, 20 ms/step).
4. **The browser draws the composite frame onto one `<canvas>`** as it
   arrives — no pre-rendered file involved, and both sides are always
   perfectly in sync since they came from the same tick.
5. **Once a side finishes** (ball lands, robot falls, or episode times out),
   it keeps advancing physics-only (no policy calls, metrics frozen at that
   moment) so it doesn't freeze mid-frame while the other side finishes. A
   shared extra tail (60 steps) after both are done lets you watch the
   ball(s) settle.
6. **When both episodes end**, the backend sends one final JSON message
   with each side's real metrics — pulled from the same `info` dict the
   eval scripts use: landing error (cm), success, whether the robot fell,
   summed reward, release time, seed. The page then shows a comparison
   table plus a headline "landing-error reduction" stat computed from the
   two.

There are also two single-policy endpoints kept around as simpler
alternatives/API surface, not wired into the current page:
- `POST /run` — runs the RL policy once, writes an MP4 to `webapp/videos/`,
  and returns its URL — useful if you want a downloadable clip (e.g. to
  embed in a slide deck) instead of a live view.
- `WS /ws/run` — the original single-policy live stream (RL only).

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
  app.py          FastAPI app: POST /run, WS /ws/run, WS /ws/compare, static mounts
  runner.py       SimulationRunner: loads envs + policy once, runs episodes
  static/
    index.html    The GUI (Audi red/black themed) — side-by-side canvas + comparison table
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
