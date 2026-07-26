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

# Task 2 - Residual PPO for Unitree G1 Ball Throwing

This branch contains the reproducible reinforcement-learning deliverables for
Audi Development Camp 2026. It is intentionally isolated from Task 1 and is
based on the frozen Task 1 contract:

- tag: `task1-baseline-v1.0`
- commit: `7a370663cbcc1aa96438dffc9f6331d3bf4ef35c`
- robot: official Unitree G1, 29 actuators
- ball radius: `0.04 m`
- target center: `(0.55, 0.00) m`
- success radius: `0.10 m`
- release time: `0.65 s`

## Result

The selected policy is a disclosed residual PPO policy around the scripted
baseline. It does not claim to learn the entire throw from zero.

Under an identical 100-seed nominal evaluation:

| Policy | Success | Mean landing error | P90 error | Falls |
|---|---:|---:|---:|---:|
| Scripted baseline | 100% | 1.150 cm | 1.223 cm | 0% |
| Selected PPO best | 100% | 0.674 cm | 0.775 cm | 0% |

The selected PPO policy reduces mean landing error by approximately 41.4%
while preserving 100% success and zero falls. The final one-million-step model
degraded; therefore the automatically preserved best policy is the selected
deliverable. Best, final, and ten checkpoints are all retained.

## Repository structure

```text
assets/       Frozen G1 MJCF, scene, and required meshes
envs/         Frozen Task 1 environment and Task 2 residual wrapper
configs/      PPO configuration used by the selected run
train/        Reproducible parallel PPO training entry point
evaluation/   Policy evaluation, comparison, and plot generation
scripts/      Snapshot verification and live PPO viewer
outputs/
  models/     Best, final, and 100k-step checkpoints
  logs/       Run metadata, Monitor data, and evaluation arrays
  plots/      Training, nominal, and robustness figures and raw metrics
docs/         Architecture, decisions, frozen hashes, and training report
tests/        Contract and environment tests
```

## Setup

Python 3.11 is recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Verify

```powershell
.\.venv\Scripts\python.exe scripts\verify_frozen_snapshot.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## View the selected PPO policy

```powershell
.\.venv\Scripts\python.exe scripts\view_ppo.py
```

## Reproduce training

The configuration uses eight CPU MuJoCo worker processes for rollout
collection and CUDA for PyTorch PPO updates.

```powershell
.\.venv\Scripts\python.exe train\train_ppo.py `
  --timesteps 1000000 `
  --run-name ppo_robust_1m_seed2026
```

Each run receives a unique directory and saves TensorBoard logs, evaluation
logs, metadata, best/final models, and 100k-step checkpoints. Existing runs
are never overwritten.

## Reproduce the comparison

```powershell
.\.venv\Scripts\python.exe evaluation\compare_baseline_ppo.py `
  --run-dir outputs\models\selected `
  --episodes 100 `
  --seed 2026
```

See [the full training report](docs/TRAINING_REPORT.md) for failed runs,
reward-design iterations, robustness results, and limitations.

Robot-model provenance and licensing are documented in
[`docs/model_source.md`](docs/model_source.md).

# Task 2 - Residual PPO for Unitree G1 Ball Throwing

This branch contains the reproducible reinforcement-learning deliverables for
Audi Development Camp 2026. It is intentionally isolated from Task 1 and is
based on the frozen Task 1 contract:

- tag: `task1-baseline-v1.0`
- commit: `7a370663cbcc1aa96438dffc9f6331d3bf4ef35c`
- robot: official Unitree G1, 29 actuators
- ball radius: `0.04 m`
- target center: `(0.55, 0.00) m`
- success radius: `0.10 m`
- release time: `0.65 s`

## Result

The selected policy is a disclosed residual PPO policy around the scripted
baseline. It does not claim to learn the entire throw from zero.

Under an identical 100-seed nominal evaluation:

| Policy | Success | Mean landing error | P90 error | Falls |
|---|---:|---:|---:|---:|
| Scripted baseline | 100% | 1.150 cm | 1.223 cm | 0% |
| Selected PPO best | 100% | 0.674 cm | 0.775 cm | 0% |

The selected PPO policy reduces mean landing error by approximately 41.4%
while preserving 100% success and zero falls. The final one-million-step model
degraded; therefore the automatically preserved best policy is the selected
deliverable. Best, final, and ten checkpoints are all retained.

## Repository structure

```text
assets/       Frozen G1 MJCF, scene, and required meshes
envs/         Frozen Task 1 environment and Task 2 residual wrapper
configs/      PPO configuration used by the selected run
train/        Reproducible parallel PPO training entry point
evaluation/   Policy evaluation, comparison, and plot generation
scripts/      Snapshot verification and live PPO viewer
outputs/
  models/     Best, final, and 100k-step checkpoints
  logs/       Run metadata, Monitor data, and evaluation arrays
  plots/      Training, nominal, and robustness figures and raw metrics
docs/         Architecture, decisions, frozen hashes, and training report
tests/        Contract and environment tests
```

## Setup

Python 3.11 is recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Verify

```powershell
.\.venv\Scripts\python.exe scripts\verify_frozen_snapshot.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## View the selected PPO policy

```powershell
.\.venv\Scripts\python.exe scripts\view_ppo.py
```

## Reproduce training

The configuration uses eight CPU MuJoCo worker processes for rollout
collection and CUDA for PyTorch PPO updates.

```powershell
.\.venv\Scripts\python.exe train\train_ppo.py `
  --timesteps 1000000 `
  --run-name ppo_robust_1m_seed2026
```

Each run receives a unique directory and saves TensorBoard logs, evaluation
logs, metadata, best/final models, and 100k-step checkpoints. Existing runs
are never overwritten.

## Reproduce the comparison

```powershell
.\.venv\Scripts\python.exe evaluation\compare_baseline_ppo.py `
  --run-dir outputs\models\selected `
  --episodes 100 `
  --seed 2026
```

See [the full training report](docs/TRAINING_REPORT.md) for failed runs,
reward-design iterations, robustness results, and limitations.

Robot-model provenance and licensing are documented in
[`docs/model_source.md`](docs/model_source.md).


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

This branch intentionally contains no CEM and no RL so it can serve as the
honest baseline for the stacked optimization branch.
