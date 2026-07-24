# Sim2Real Robustness Testing

Author: Tianyu Yao (Sim2Real Lead)

## Prerequisites

Python 3.10+ with these packages:

```bash
pip install mujoco gymnasium numpy stable-baselines3 torch
```

## Quick Start

### 1. Clone the repo and switch to this branch

```bash
git clone https://github.com/aim-t/audi-ai-robotics-2026.git
cd audi-ai-robotics-2026
git checkout feature/sim2real
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. View the trained robot (no perturbations)

```bash
python scripts/play_robustness.py
```

This opens a MuJoCo window. Close it to exit.

### 4. Compare clean vs noisy side-by-side

```bash
python scripts/compare_side_by_side.py
```

Opens two MuJoCo windows: left = clean, right = 7 perturbations active.

### 5. Run full evaluation (numbers only, no GUI)

```bash
python scripts/evaluate_robustness.py --episodes 100
```

## Files

| File | Purpose |
|------|---------|
| `envs/g1_robustness_env.py` | Environment with 7 configurable perturbations |
| `scripts/evaluate_robustness.py` | Numerical evaluation (clean vs noisy) |
| `scripts/play_robustness.py` | Single-window visual viewer |
| `scripts/compare_side_by_side.py` | Side-by-side clean/noisy viewer |
| `outputs/robustness_final.json` | 100-episode evaluation data |
| `outputs/per_param_results.json` | Per-parameter isolation test data |
| `outputs/per_param_chart.png` | Bar chart of per-parameter impact |
| `outputs/landing_error_cdf.png` | CDF of landing error distribution |

## Dependencies on feature/rl branch

This branch builds on the RL team's work. The following files from `feature/rl` are required and already included:
- `envs/ppo_throw_env.py`
- `envs/g1_fixed_body_throw_env.py`
- `assets/scene_throw.xml`
- `assets/g1.xml` + STL mesh files
- `outputs/models/selected/best/best_model.zip` (trained policy)
