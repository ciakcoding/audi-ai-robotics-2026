# Architecture

The repository has three layers:

1. `assets/` and `envs/g1_fixed_body_throw_env.py` are a frozen copy of the
   accepted Task 1 world model and evaluation truth.
2. `envs/ppo_throw_env.py` adds only the Task 2 learning layer: seven bounded
   residual corrections around the disclosed scripted arm trajectory.
3. `train/`, `evaluation/`, and `outputs/` contain the reproducible PPO
   workflow and evidence.

Task 2 does not import from a mutable Task 1 checkout. The core files are
guarded by SHA-256 values in `docs/frozen_snapshot.json`.
The published scene removes only obsolete commented-out XML from the frozen
source; the active MuJoCo model and compiled physics are unchanged.

The scripted release time remains `0.65 s`; PPO changes arm commands only.
Reward guides optimization, while success, landing error, first-contact
detection, and fall metrics remain defined by the frozen environment.

Eight independent MuJoCo environments collect rollouts in CPU subprocesses.
Stable-Baselines3 aggregates the rollout and performs PPO network updates on
CUDA. Evaluation uses a separate environment and deterministic actions.
