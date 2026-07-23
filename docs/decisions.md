# Engineering decisions

## PPO

PPO was selected because the teacher recommends it as a robust, explainable
default for continuous control with mature Stable-Baselines3 support.

## Residual policy

The scripted baseline was already stable and highly accurate. Following the
teacher's hybrid recommendation, PPO learns bounded corrections instead of
relearning the entire throw. This lowers action complexity and preserves an
explainable reference.

## Model retention

Training was non-monotonic. The best policy occurred near 100k steps, while
the final policy degraded badly. Therefore every run retains:

- the best deterministic evaluation policy;
- the final policy;
- checkpoints every 100k steps;
- evaluation arrays, Monitor logs, metadata, and wall time.

## Honest comparison

The first two one-million-step runs did not beat the baseline and remain in
the local archived research record. Reward was revised to align with landing
precision. The selected third run improves mean error but still has a larger
single robustness outlier, documented in `TRAINING_REPORT.md`.
