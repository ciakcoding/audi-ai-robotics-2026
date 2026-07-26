# Audi AI Robotics 2026

Audi Development Camp 2026 — training a Unitree G1 humanoid (29 DoF) in
MuJoCo to perform two tasks, each built up in three stages: a scripted
baseline, a residual reinforcement-learning policy on top of it, and a
Sim2Real domain-randomization robustness check.

| | Task | Baseline | RL |
|---|---|---|---|
| **Level 02** | Throw a ball to a fixed ground target while balancing (no locomotion) | 100% success, 1.150 cm mean landing error | 100% success, 0.674 cm mean landing error (−41% error) |
| **Level 03** | Walk ~2.2 m to a basketball hoop and shoot | scripted CEM-derived shot (293/300 direct success) | one-decision residual over 15 shot parameters (297/300 direct success, 3.336 cm mean hoop-plane error) |

Both policies are also stress-tested under domain randomization (joint
friction/damping, actuator gain, floor friction, contact stiffness, observation
noise, control latency) to check they don't just memorize the nominal
simulator.

- **Trello board**: https://trello.com/b/z42I2vSE
- **Final report**: https://docs.google.com/document/d/1HUYO_wSMAxBY-qL1g_YRi02oJLaMqBbRzlGysW7D7M0/edit?tab=t.28q7u0wql8m3

## Quick start: watch it run

The fastest way to see everything working is the local web GUI — click a
button, watch the trained policies run live in the browser, no terminal
interaction needed after startup:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn webapp.app:app --reload
```

Then open `http://localhost:8000` (Level 02) and
`http://localhost:8000/level03.html` (Level 03). Full details on what each
page is doing under the hood: [`webapp/README.md`](webapp/README.md).

## How it's built

Each level is a stack of three layers, each frozen before the next is built
on top so later work can't quietly change what came before:

1. **Scripted baseline** — a hand-authored, non-learning controller (fixed
   arm keyframes / motion trajectory) that solves the task directly. This is
   the disclosed, auditable reference every later stage is measured against.
2. **Residual RL policy** — a PPO (Level 02) or SAC→PPO (Level 03) policy
   that predicts a *bounded correction* on top of the scripted baseline,
   rather than learning the task from scratch. Level 02's policy corrects
   the arm swing every control step; Level 03's policy makes one decision
   per episode (a residual over 15 expert trajectory parameters), since the
   shot motion itself is a single planned arc.
3. **Sim2Real robustness check** — the same trained RL policy re-run under
   randomized physics (friction, damping, actuator strength, contact
   softness) and sensor/actuation noise, to see how much performance holds
   up outside the exact conditions it was trained in.

Each stage's environment/model files are treated as frozen contracts — see
[`docs/architecture.md`](docs/architecture.md) and
[`docs/frozen_snapshot.json`](docs/frozen_snapshot.json) for how that's
enforced for Level 02.

## Repository structure

```text
assets/               G1 MJCF model, scenes, and meshes (Menagerie Unitree G1)
envs/                 Level 02 Gymnasium envs (baseline, PPO residual, robustness/Sim2Real)
scripts/              Scripted baselines, live viewers, evaluation entry points
train/                Level 02 PPO training entry point
evaluation/            Level 02 policy evaluation and baseline-vs-RL comparison
training_extension/   Level 03: derived baseline, CEM expert trajectories, RL training/eval
outputs/               Trained models, logs, plots, evaluation artifacts (mostly gitignored)
webapp/                FastAPI + HTML/JS/canvas GUI serving both levels' live demos
docs/                  Architecture, decisions, frozen hashes, training report
tests/                 Contract and environment tests
Dockerfile, fly.toml   Headless deployment for the web GUI
```

## Running things directly (without the GUI)

Level 02:
```bash
python scripts/view_baseline.py                 # scripted baseline, looping
python scripts/view_ppo.py                       # trained RL policy
python evaluation/compare_baseline_ppo.py --episodes 100 --seed 2026
python -m unittest discover -s tests -v
```

Level 03 (see [`training_extension/README.md`](training_extension/README.md)
and [`training_extension/RL_README.md`](training_extension/RL_README.md) for
the full picture):
```bash
python -m training_extension.view_derived_baseline
python -m training_extension.evaluate_ppo_parameters \
  --model training_extension/frozen/ppo_parameters_12288_selected_20260726/selected_model.zip
```

Sim2Real (Level 02): see [`SIM2REAL_README.md`](SIM2REAL_README.md).
Sim2Real (Level 03): `scripts/level_3_view_noisy.py` /
`scripts/level_3_evaluate_robustness.py`.

Windows: `run_baseline.bat` / `run_baseline.ps1` wrap the equivalent commands
with `.venv\Scripts\python.exe` and PowerShell syntax.

## Deployment

The web GUI ships as a headless Docker image (`MUJOCO_GL=osmesa`, no GPU
needed) with a ready-made `fly.toml` for Fly.io. See the
[Deployment section of `webapp/README.md`](webapp/README.md#deployment) for
build/run/deploy commands and cost caveats.

## Further reading

- [`docs/architecture.md`](docs/architecture.md), [`docs/decisions.md`](docs/decisions.md), [`docs/TRAINING_REPORT.md`](docs/TRAINING_REPORT.md) — Level 02 design and training report
- [`docs/model_source.md`](docs/model_source.md) — robot model provenance and licensing
- [`training_extension/README.md`](training_extension/README.md), [`training_extension/RL_README.md`](training_extension/RL_README.md) — Level 03 baseline and RL extension
- [`SIM2REAL_README.md`](SIM2REAL_README.md) — Level 02 robustness testing
- [`webapp/README.md`](webapp/README.md) — web GUI internals and deployment
