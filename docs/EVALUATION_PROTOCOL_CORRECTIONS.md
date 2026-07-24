# Evaluation protocol corrections

## Purpose

This document resolves conflicts between the draft *Success Criteria &
Evaluation Metrics* document and the frozen, validated project contract. All
new evaluation data must use the project values below. The Task 1 world model
and the trained PPO policy are not changed.

## Authoritative project values

| Item | Authoritative value |
|---|---|
| Robot | Unitree G1 29DoF (`g1.xml`), floating base |
| Ball | Sphere, radius `0.04 m`, mass `0.05 kg` |
| Target center | `(0.55, 0.00, 0.00) m` |
| Success radius | `0.10 m` |
| Landing event | First ball-floor contact |
| Physics timestep | `0.002 s` |
| Control timestep | `0.02 s` (10 physics substeps) |
| Gravity | `(0, 0, -9.81) m/s²` |
| Release configuration | Scripted at `0.65 s`; observed on the control grid at approximately `0.66 s` |
| Episode duration | `1.8 s` |
| Fall definition | Torso height `<0.60 m` or torso tilt `>45°` |
| Task 1 version | commit `7a370663cbcc1aa96438dffc9f6331d3bf4ef35c`, tag `task1-baseline-v1.0` |
| Learned policy | Selected PPO `best_model.zip`, not the degraded final checkpoint |

## Corrections to the earlier draft

1. **Ball size must be unambiguous.** “Ball (4 cm)” is replaced by “ball
   radius 4 cm (diameter 8 cm), mass 50 g.”
2. **The success radius is 0.10 m, not 0.15 m.** The draft states a 0.10 m
   target radius but later uses `d_err <= 0.15 m`. All headline hit rates and
   comparisons use the stricter frozen 0.10 m radius.
3. **Release timing is not 0.68 s.** The frozen configuration is 0.65 s.
   Because controls update every 0.02 s, the recorded event is normally 0.66 s.
4. **The episode is not 5 s.** The frozen task episode is 1.8 s. A separate
   3 s post-contact continuation may be used only to measure rolling and
   recovery diagnostics; it does not change headline success.
5. **Air density 1.2 is not enabled.** Adding aerodynamic drag would change the
   world model and invalidate direct comparison with the trained policy.
   Therefore the evaluator records air drag as disabled.
6. **The 2° and 5° tilt values were not part of the frozen success contract.**
   They are reported as diagnostics, not used to silently redefine success.
   The authoritative safety gate remains the frozen fall definition.
7. **“Final policy” must not mean the last checkpoint.** The 1M-step final
   checkpoint degraded. The selected best checkpoint is the formal learned
   policy; best, final, and periodic checkpoints remain preserved for audit.
8. **Impact and rolling metrics require explicit instrumentation.** They must
   be measured from MuJoCo state/contact data, not estimated from video.
9. **A weighted reward is not an evaluation metric.** Reward is retained only
   as a training diagnostic. Headline comparison uses fall rate, success rate,
   and first-contact landing error under identical seeds.
10. **Evaluation size is 100 episodes per policy.** This exceeds the draft
    minimum of 20 and provides stronger statistics. Both policies use seeds
    `2026-2125`.

## Added reproducible data

The extended evaluator produces one raw row per policy and episode with:

- seed, policy name, and selected checkpoint;
- first-contact XY and first-contact error;
- final XY after a 3 s post-contact diagnostic continuation;
- rolling drift and final rolling error;
- release time;
- maximum hold tilt, tilt at release, and maximum episode tilt;
- time to a sustained 2° diagnostic stability window, when reached;
- official fall flag and post-contact diagnostic fall flag;
- downward impact speed and peak floor normal force;
- invalid-flight contacts and invalid reason;
- five stage diagnostics;
- reward and wall-clock duration.

Files:

- `evaluation/evaluate_extended_metrics.py`
- `outputs/extended_evaluation/extended_episode_metrics.csv`
- `outputs/extended_evaluation/extended_summary.json`

Run both nominal and authorized `+/-0.08 rad` protocols from the repository
root:

```powershell
.\run_extended_evaluation.bat .\.venv\Scripts\python.exe
```

Alternatively, run one protocol directly:

```powershell
python evaluation\evaluate_extended_metrics.py --episodes 100 --seed 2026 --joint-noise 0.08 --output-dir outputs\extended_evaluation\joint_noise_0p08
```

## Comparison rule

A learned policy may be called better only when:

1. its official fall rate is no greater than the baseline fall rate; and
2. under the same seeds and frozen 0.10 m target, it has a higher success rate,
   or, when success is tied, a lower mean first-contact landing error.

No world-model parameter, target definition, or success threshold is changed
between the baseline and PPO evaluation.
