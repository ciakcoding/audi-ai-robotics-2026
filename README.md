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
