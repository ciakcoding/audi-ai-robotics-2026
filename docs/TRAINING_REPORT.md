# Task 2 PPO training report

## Frozen dependency and method

- Frozen Task 1 tag: `task1-baseline-v1.0`
- Frozen commit: `7a370663cbcc1aa96438dffc9f6331d3bf4ef35c`
- Algorithm: Stable-Baselines3 PPO with `MlpPolicy`
- Collection: 8 parallel CPU MuJoCo environments
- Updates: CUDA
- Method: disclosed residual PPO around the scripted baseline
- Release time, target, success radius, physics, and final metrics remain frozen

The hybrid method follows the teacher's recommendation to script first and
learn corrections around a stable baseline. It is not described as learning
from zero.

## Completed runs

Three separate one-million-step runs were completed. Every run preserves ten
100k-step checkpoints, a best evaluation policy, a final policy, TensorBoard
logs, evaluation logs, metadata, and wall time.

1. `ppo_residual_1m_seed2026_20260723`: initial reward design. PPO did not beat
   the baseline.
2. `ppo_precision_1m_seed2026_20260723`: precision-aligned reward. Best PPO was
   close to, but still worse than, the baseline.
3. `ppo_robust_1m_seed2026_20260723`: precision reward plus explicitly recorded
   initial-joint-noise training. This produced the selected best policy.

The failed/intermediate runs are retained as evidence of reward iteration and
honest interpretation.

## Fair nominal comparison

Protocol: 100 episodes, seeds 2026-2125, target `(0.55, 0.00)`, success radius
`0.10 m`, release time `0.65 s`, and the frozen Task 1 reset distribution.

| Policy | Success | Mean error | P90 error | Max error | Falls |
|---|---:|---:|---:|---:|---:|
| Baseline | 100% | 1.150 cm | 1.223 cm | 1.305 cm | 0% |
| PPO best | 100% | 0.674 cm | 0.775 cm | 0.818 cm | 0% |
| PPO final | 16% | 22.088 cm | 35.650 cm | 41.658 cm | 0% |

The selected best PPO reduces mean nominal landing error by approximately
41.4% while retaining the baseline's 100% success and zero-fall result.

## Robustness comparison

The same 100 seeds were repeated with an additional, identical `±0.08 rad`
initial right-arm joint perturbation for both policies. No physics or scoring
parameter changed.

| Policy | Success | Mean error | P90 error | Max error | Falls |
|---|---:|---:|---:|---:|---:|
| Baseline | 100% | 1.179 cm | 1.439 cm | 1.515 cm | 0% |
| PPO best | 100% | 0.754 cm | 0.971 cm | 5.477 cm | 0% |
| PPO final | 10% | 20.927 cm | 32.557 cm | 42.464 cm | 0% |

Mean error improves by approximately 36.0%. PPO best has a better mean and P90
but a worse single maximum outlier, which must be reported as a limitation.

## Model selection conclusion

Use `outputs/models/selected/best/best_model.zip` for evaluation and demo. Do
not use the final model. Later training degraded substantially, which is why
best, final, and periodic checkpoints are all preserved.
