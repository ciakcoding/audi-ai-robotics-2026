# Fixed-target PPO training-progress checkpoints

These checkpoints are the exact policies used in the training-progress video.
They are separate from the selected 1M-step robust run.

Evaluation protocol:

- fixed target: `(0.55, 0.00) m`;
- success radius: `0.10 m`;
- 100 deterministic episodes per checkpoint;
- seeds: `50000-50099`;
- additional right-arm initial joint noise: `+/-0.08 rad`;
- frozen Task 1 commit: `7a370663cbcc1aa96438dffc9f6331d3bf4ef35c`.

| Steps | File | Success | Mean landing error | Mean reward |
|---:|---|---:|---:|---:|
| 0 | `initial_model.zip` | 100% | 1.168 cm | 65.74 |
| 25,000 | `checkpoints/ppo_throw_25000_steps.zip` | 100% | 1.052 cm | 66.32 |
| 50,000 | `checkpoints/ppo_throw_50000_steps.zip` | 100% | 0.892 cm | 67.11 |
| 75,000 | `checkpoints/ppo_throw_75000_steps.zip` | 100% | 0.858 cm | 67.20 |
| 100,000 | `checkpoints/ppo_throw_100000_steps.zip` | 100% | 0.708 cm | 67.94 |

Use these files to reproduce the learning-progression montage/video. Use
`outputs/models/selected/best/best_model.zip` for the formal robust-policy
comparison.
