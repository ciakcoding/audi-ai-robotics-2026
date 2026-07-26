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

A small local website with two side-by-side live comparisons, no terminal
required:
- **Baseline vs RL Policy** — click **Run Simulation** and watch the
  scripted Task 1 baseline and the trained Task 2 RL policy throw the ball
  from the same starting conditions — baseline left, RL right.
- **Sim2Real Robustness Check** — click **Run Sim2Real Test** and watch
  the *same* trained RL policy run once clean and once under the full
  Task 3 domain-randomization gauntlet — nominal left, Sim2Real right.

Both end in a metrics comparison table.

## What's actually happening (Baseline vs RL Policy)

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

## What's actually happening (Sim2Real Robustness Check)

Same mechanics as above (`WS /ws/sim2real`, same lockstep-and-composite
approach via the shared `SimulationRunner._stream_pair()` helper), but both
sides run the **same** trained RL policy — the only difference is the
environment:
- **Nominal (left):** `WebRobustnessEnv(enable_all=False)` — identical
  physics to the Baseline-vs-RL page's RL side, just wrapped in the
  robustness env class for a consistent interface.
- **Sim2Real (right):** `WebRobustnessEnv(enable_all=True)` — turns on all
  7 of `envs/g1_robustness_env.py`'s (`G1RobustnessEnv`) perturbations at
  once: observation noise, joint friction/damping randomization, floor
  friction randomization, actuator gain randomization, contact
  stiffness/impedance randomization, control latency, and target-position
  noise. This is a live version of what `scripts/evaluate_robustness.py`
  does over 50-100 episodes and averages — here you watch one draw at a
  time. Per the project's own `outputs/per_param_results.json`, target-
  position noise is by far the dominant contributor (mean best-distance
  ~0.042 m alone vs. ~0.02 m clean; the other 6 perturbations are each
  individually negligible), so the most visible effect is usually the green
  target sphere sitting in a slightly different spot on the right.
- The domain-randomization draw itself (`G1RobustnessEnv.reset()`'s
  `np.random.uniform`/`normal` calls) is **not** tied to the seed passed to
  `env.reset(seed=...)` — it uses bare global `np.random`, exactly like the
  original `G1RobustnessEnv`/`evaluate_robustness.py` do. So the nominal
  side's initial arm noise is reproducible per seed, but the randomization
  strength on the Sim2Real side varies every run, by design — that's the
  whole point of watching several runs.
- `WebRobustnessEnv` exists for the same reason `WebPPOThrowEnv` does:
  `G1RobustnessEnv.__init__` calls `PPOThrowEnv.__init__`, which hardcodes
  the broken top-level scene path. Since `G1RobustnessEnv` always inherits
  directly from `PPOThrowEnv`, there's no cooperative-`super()` way to
  splice in the working path, so `WebRobustnessEnv.__init__` in
  `webapp/runner.py` copies `G1RobustnessEnv.__init__`'s body verbatim,
  swapping only the one `super().__init__()` call — see the docstring
  there if `envs/g1_robustness_env.py` ever changes and this needs
  re-syncing.

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

## Level 03 — Basketball-Hoop Shot (`level03.html`, `webapp/runner_level03.py`)

A second, separate page at `/level03.html` (not linked from `index.html` —
Level 02's page is intentionally untouched). Unlike Level 02, this isn't a
lockstep comparison in a shared environment: **each side runs its own
script's actual simulation, exactly as written, in its own scene**, because
no unified Level 03 env exists in this project to make a true
apples-to-apples comparison meaningful.

- **Baseline (left)**: `scripts/view_baselines_LEVEL03_v031!.py`'s own
  `G1FixedBodyThrowEnv` + `OptionDBasketballPolicy`, dynamically loaded via
  `importlib` (the same technique `training_extension/derived_baseline.py`
  already uses for this exact file, since the `!` in the filename isn't a
  valid Python module name). `webapp/runner_level03.py`'s `BaselineSlot`
  replays that script's `view_baseline()` per-step body line-for-line — the
  keyframe policy, the pelvis anti-drift gyro correction, the release at
  policy step 406, the hoop-crossing and rim-impact-force detection — just
  swapping its interactive `mujoco.viewer.launch_passive()` loop for
  headless stepping and offscreen rendering. Target is 1.8m out; the loop
  always runs the full fixed 850 steps, exactly like the script (no
  early-exit condition exists in it).
- **RL (right)**: `training_extension/view_ppo_parameters.py`'s own
  trained PPO model + `VecNormalize` over `SACShotParameterEnv` — a
  one-decision parameter-residual policy: called exactly once per episode,
  from the initial observation, predicting a residual over 15 expert shot
  parameters (arm load/release angles + release timing), which are then
  played out for the whole walk-dip-throw sequence via
  `training_extension/optimize_direct.py`'s `controller_action()` — not a
  per-step neural network call. Target is 2.2m out, and
  `BasketballResidualEnv` terminates early
  (~426 steps) once the ball crosses the hoop plane, touches the backboard,
  or the robot falls. `webapp/runner_level03.py`'s `RLSlot` then holds it in
  a stabilized idle pose (`BasketballResidualEnv._apply_peer_stabilizer()`,
  the same anti-drift torques it applies every substep during real
  stepping) for the remainder of the 850-step window, so it doesn't
  collapse once its episode ends and so both panels run for the same
  visual duration as the baseline's fixed-length loop.
- **Metrics don't share a schema.** The baseline script computes
  `crossed_hoop`/hoop-crossing speed/rim-impact force/max torso tilt/final
  distance; the RL env computes `success`/crossing XY error/backboard
  contact/fall/airborne distance/release step. The page shows both sets of
  native numbers side by side, under clearly separated headers, rather than
  forcing them into shared rows that would misrepresent one side's scoring
  as the other's.
- **`assets/g1.xml`'s meshdir was fixed** (`meshdir="assets"` →
  `meshdir="."`) to make `assets/scene_throw_LEVEL03.xml` loadable at all —
  the same broken-mesh-path bug Level 02's webapp worked around, but this
  time there was no existing self-contained alternative scene to point at
  instead, so the shared asset itself was corrected. This also happens to
  fix the top-level `assets/scene_throw.xml` Level 02 originally couldn't
  load (see below) — a side effect, not something exercised by either page.
- **~25-30s per click.** 850 steps × 0.02s = 17s of simulated time, plus
  render/JPEG-encode overhead. There's no way to shorten this without
  deviating from the baseline script's own fixed loop length.

### Level 03 Sim2Real Robustness Check

A second comparison on the same page (`WS /ws/level03/sim2real`), same
spirit as Level 02's Sim2Real section but built from
`scripts/level_3_view_noisy.py` instead: the **same** trained RL policy run
once clean (`RLSlot`, nominal) and once under `NoisyRLSlot`'s domain
randomization — joint friction/damping (0.7-1.3x), actuator force range
(0.85-1.0x), contact stiffness/impedance solref/solimp (0.5-2.0x), floor
friction (0.5-1.5x), and ±3cm target-position noise on x/y only — using
that script's own perturbation ranges verbatim (which happen to match
Level 02's `G1RobustnessEnv._enable_all_defaults()` ranges). Unlike the
baseline-vs-RL comparison above, both sides here are `BasketballResidualEnv`
instances with the same scoring contract, so the table shows one shared
set of rows, not two separate schemas.

Matching that script exactly: the observation used for the model's one-shot
residual prediction is captured *before* the perturbations are applied (the
script calls `vector_env.reset()`, then perturbs physics/target, then
predicts from the pre-perturbation observation) — reproduced here even
though it means the residual is predicted without seeing the noise it's
about to be tested against. The randomization draw itself uses bare
`np.random`, not the seeded env RNG, so — exactly like Level 02's Sim2Real
page — the nominal side's initial state is reproducible per seed but the
randomization strength on the Sim2Real side varies every run by design.
Both sides terminate early (~426 steps typical, well under
`BasketballResidualEnv.max_policy_steps`'s 1100-step hard cap), so this one
breaks out once both sides are done plus a short 40-step settle tail,
rather than running the baseline comparison's fixed 850.

## Files

```text
webapp/
  app.py             FastAPI app: POST /run, WS /ws/run, /ws/compare,
                      /ws/sim2real, /ws/level03/compare, /ws/level03/sim2real,
                      static mounts
  runner.py          Level 02 SimulationRunner: loads envs + policy once, runs episodes
  runner_level03.py  Level 03: BaselineSlot (scripts/view_baselines_LEVEL03_v031!.py,
                      loaded via importlib), RLSlot (view_ppo_parameters.py's model),
                      NoisyRLSlot (level_3_view_noisy.py's Sim2Real gauntlet)
  static/
    index.html    Level 02 GUI (Audi red/black themed) — two side-by-side canvases + tables
    level03.html  Level 03 GUI, same visual format — two side-by-side canvases + tables
  videos/         Generated MP4s from POST /run (gitignored)
```

## Run it

```bash
source .venv/bin/activate
uvicorn webapp.app:app --reload
```

Open `http://localhost:8000` for Level 02, then click **Run Simulation**
(top section) or **Run Sim2Real Test** (bottom section). Open
`http://localhost:8000/level03.html` for Level 03 (no link between the two
pages yet) and click **Run Simulation** — expect ~25-30s per run.

`MUJOCO_GL` controls the rendering backend and needs no code changes:
- Local machine with a display (default here): unset, or `glfw`.
- Headless server, CPU only: `MUJOCO_GL=osmesa`.
- Headless server with GPU: `MUJOCO_GL=egl`.

## Known limitations (v1)

- Single shared MuJoCo environment/renderer instance per page (Level 02 and
  Level 03 have separate locks, so the two pages don't block each other,
  but concurrent runs *within* one page's endpoints are serialized behind
  that page's lock, not parallelized).
- No nav link between `index.html` and `level03.html` yet — open
  `/level03.html` directly.
- No deployment/containerization yet (`Dockerfile` is a planned next step).
