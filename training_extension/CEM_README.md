# LEVEL03 CEM trajectory optimization

This branch is stacked on `feature/simulation03-derived-baseline`. It adds
cross-entropy method (CEM) trajectory optimization without changing the world,
target or scoring contract defined by the baseline branch.

**CEM is trajectory optimization, not reinforcement learning.** The selected
`state.json` stores 15 open-loop residual parameters: load/release offsets for
the two arms and waist, plus release timing. The complete scripted walk, dip,
leg extension, throw and recovery remain in the derived baseline.

## Selected result

Selected artifact:

`cem_artifacts/selected/state.json`

Frozen validation:

- success: `20/20`
- mean ball-center crossing error: `0.03030 m`
- direct shots: `20/20`
- backboard contacts: `0`
- falls: `0`

Fresh 100-seed validation after rebasing on teammate `v031`
(`50000..50099`):

- success/direct shots: `98/100`
- mean ball-center crossing error: `0.03635 m`
- maximum crossing error: `0.36735 m`
- backboard contacts: `0`
- falls: `0`

The two misses are retained in
`cem_artifacts/selected/evaluation100_seed50000/episodes.json`; the stricter
100-seed result is not rounded up to 100%.

No-CEM baseline reference on the parent branch:

- success: `0/100`
- mean hoop-plane crossing error: `1.1032 m`
- backboard contacts: `0`
- falls: `0`

## Reproduce without rerunning optimization

From the repository root:

```powershell
python -m pip install -r training_extension/requirements-cem.txt
python -m training_extension.view_direct
```

Validate the saved parameters:

```powershell
python -m training_extension.validate_direct `
  --state training_extension/cem_artifacts/selected/state.json `
  --episodes 100 `
  --seed 50000 `
  --workers 8 `
  --output outputs/cem_eval100_seed50000
```

Run code/state checks:

```powershell
python -m unittest discover -s training_extension/tests -v
python -m training_extension.quality_check_cem --smoke-episodes 2
```

## Rerun optimization

The saved result is sufficient for playback and validation. To run a new CEM
search:

```powershell
python -m training_extension.optimize_direct `
  --iterations 40 `
  --population 32 `
  --seed-count 4 `
  --workers 8 `
  --output outputs/new_cem_run
```

Optimization is CPU-parallel; it does not require CUDA.

## Milestones

`cem_artifacts/milestones/` contains curated unsuccessful and intermediate
states. They are retained to document why numerical success alone was
insufficient:

| Stage | Outcome | Decision |
|---|---|---|
| `01_invalid_hand_in_hoop` | high numerical success by moving the hand near the hoop | rejected as cheating |
| `02_airborne_low_release` | airborne direct flight but low sweeping release | rejected as unnatural |
| `03_high_release_onehand` | higher release, guide hand too far away | continued |
| `04_twohand_gap_too_wide` | two-hand attempt, separation still above limit | continued |
| `05_twohand_strict_success` | strict two-hand release at the nearer target | accepted intermediate |
| `07_far_target_openloop` | farther target and visible ball flight | footwork still needed work |
| `08_second_step_fix` | corrected second-step landing | post-shot arms still crossed |
| `09_postshot_separation` | separated recovery lanes | guide hand still refined |
| selected v17 | no crossing arms, direct far throw | selected |

No PPO, TD3 or SAC model, replay buffer, TensorBoard log or RL training source
is included in this branch.
