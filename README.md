# audi-ai-robotics-2026
Audi Development Camp 2026 - AI Robotics Team

## Task 1 scripted baseline

This baseline uses the official MuJoCo Menagerie Unitree G1 `g1.xml` model
with 29 actuators. The ball radius is 4 cm, the target center is
`(0.55, 0.00)`, and the unchanged success radius is 10 cm.

### Setup

From this repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Python 3.10-3.12 is recommended. All required G1 XML and mesh assets are
included in `assets/`.

### Run

View the continuously looping scripted non-learning baseline:

```powershell
.\run_baseline.ps1
```

On Windows, `run_baseline.bat` can also be double-clicked from Explorer or
the VS Code file tree after the environment has been installed.

Evaluate 100 deterministic episodes:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_baseline.py --episodes 100 --seed 2026
```

Run contract tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Evaluation results are generated under the ignored `outputs/` directory.

Validated with MuJoCo 3.10: 100/100 successful episodes from seed 2026,
mean first-contact landing error 1.15 cm, and zero detected falls.

## LEVEL03 derived basketball baseline

The independently runnable LEVEL03 derived baseline is documented in
[`training_extension/README.md`](training_extension/README.md). It preserves
the teammate `v031` implementation, overrides selected motion keyframes in a
separate policy, and evaluates a physical hoop at `(2.2, 0.0, 1.2)`.

The parent branch `feature/simulation03-derived-baseline` intentionally
contains no CEM and no RL. This stacked branch adds CEM artifacts documented
in [`training_extension/CEM_README.md`](training_extension/CEM_README.md),
while still excluding all reinforcement-learning work.

## LEVEL03 reinforcement learning

The stacked `feature/rl-on-lv3` branch adds PPO parameter-residual training,
evaluation, playback, the selected model, and compact failed/intermediate
experiment evidence. See
[`training_extension/RL_README.md`](training_extension/RL_README.md).

The selected policy was trained for 12,288 cumulative complete-shot episodes.
Across the fixed 300-seed evaluation it records 297/300 direct successes,
3.336 cm mean hoop-plane error, 31.255 cm maximum error, no backboard
contacts, and no falls. The world, hoop target, 10 cm success radius, and
anti-cheating constraints remain unchanged.
