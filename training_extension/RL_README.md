# LEVEL03 reinforcement-learning extension

This directory adds reinforcement learning on top of the separately reviewed
derived-baseline and CEM branches. It does not replace or edit the teammate's
`scripts/view_baselines_LEVEL03_v031!.py`.

## Selected method

The final policy uses Stable-Baselines3 PPO. Once per episode, it predicts a
small residual over 15 CEM trajectory parameters. The resulting trajectory is
still executed through the existing smooth interpolation, preserving the
two-step approach, dip, leg extension, two-hand lift, hand-held release,
right-hand follow-through, and separated recovery.

The frozen CEM v17 trajectory is the expert center, not an RL result. PPO is
trained from MuJoCo rewards and saves a deployable policy/value network.

## Fixed task contract

- target center: `(2.2, 0.0, 1.2)`;
- physical ball-center success radius: `0.10 m`;
- the ball must descend through the hoop plane;
- release ball-to-hoop distance must be at least `1.10 m`;
- release pelvis-to-hoop distance must be at least `1.20 m`;
- airborne horizontal distance must be at least `1.00 m`;
- nearest hand-to-hoop distance must be at least `0.45 m`;
- backboard contact, falling, dunking, hand-in-hoop behavior, and root
  translation shortcuts cannot count as success.

The world model, hoop, target, and scoring contract are unchanged from the CEM
parent branch.

## Training lineage

| Stage | Algorithm | Outcome |
|---|---|---|
| TD3 residual | frame-level TD3 | collapsed despite raw, zero-init, and filtered variants |
| PPO residual | frame-level PPO | stable, but less accurate than the expert center |
| SAC parameters | episode-level SAC | improved tail error but worsened mean error |
| PPO 1,024 | episode-level parameter PPO | first positive RL checkpoint |
| PPO 4,096 | conservative PPO fine-tune | previous selected checkpoint |
| PPO 10,752 | continued PPO | improved success, mean, and tail error |
| **PPO 12,288** | low-rate PPO refinement | **final selected checkpoint** |

The selected lineage contains 12,288 complete-shot episodes. During the final
work session, two branches of 8,192 and 4,096 additional episodes were run and
every 512-episode checkpoint was saved. Checkpoint selection used a common
100-seed sweep followed by a fixed 300-seed confirmation. The last checkpoint
was not selected merely because it was last.

Policy updates ran on an NVIDIA RTX 4060 Laptop GPU while eight CPU MuJoCo
subprocesses collected trajectories. Low GPU utilization is expected for the
small MLP because physics simulation is the bottleneck. A 16-environment
Windows launch failed before training with `BrokenPipeError/EOFError`, so the
validated eight-environment configuration was retained.

## Fixed 300-seed results

| Policy | Direct success | Mean error | Maximum error | Backboard | Falls |
|---|---:|---:|---:|---:|---:|
| CEM v17 | 293/300 | 3.865 cm | 38.514 cm | 0 | 0 |
| PPO 1,024 | 294/300 | 3.701 cm | 38.956 cm | 0 | 0 |
| PPO 4,096 | 294/300 | 3.545 cm | 39.141 cm | 0 | 0 |
| **PPO 12,288** | **297/300** | **3.336 cm** | **31.255 cm** | **0** | **0** |

Compared with PPO 4,096, the selected model adds three successes, reduces mean
error by about 5.9%, and reduces maximum error by about 20.1%. Compared with
CEM v17, mean error falls by about 13.7% and success increases by four.

## Reproduce

Use Python 3.11 in an ASCII-only checkout path on Windows:

```powershell
python -m pip install -r training_extension/requirements.txt
python -m unittest discover -s training_extension/tests -v
python -m training_extension.quality_check_rl
python -m training_extension.evaluate_ppo_parameters `
  --model training_extension/frozen/ppo_parameters_12288_selected_20260726/selected_model.zip `
  --vecnormalize training_extension/frozen/ppo_parameters_12288_selected_20260726/selected_vecnormalize.pkl `
  --episodes 300 --seed 100000 --device cpu `
  --output outputs/ppo_12288_eval300
```

Interactive playback:

```powershell
python -m training_extension.view_ppo_parameters `
  --model training_extension/frozen/ppo_parameters_12288_selected_20260726/selected_model.zip `
  --vecnormalize training_extension/frozen/ppo_parameters_12288_selected_20260726/selected_vecnormalize.pkl
```

Playback now continues for 10 seconds after the normal hoop-crossing scoring
terminal. The console reports falls before crossing and during recovery
separately. Override the extension with `--post-shot-seconds`.

Presentation/web metrics matching the names used by the teammate's v031
baseline viewer:

```powershell
python -m training_extension.export_final_web_metrics
```

This writes `training_extension/artifacts/lv3_final_web_metrics.json`. Its
`primary_display_metrics` block contains hoop-crossing speed, maximum rim
impact force, maximum torso tilt (pitch/roll/yaw), and final
ball-to-target distance. The previous RL-only values remain available in
`legacy_rl_metrics`.

Extended recovery evaluation:

```powershell
python -m training_extension.evaluate_ppo_recovery `
  --model training_extension/frozen/ppo_parameters_12288_selected_20260726/selected_model.zip `
  --vecnormalize training_extension/frozen/ppo_parameters_12288_selected_20260726/selected_vecnormalize.pkl `
  --episodes 300 --seed 100000 --post-shot-seconds 10 --workers 8 `
  --output outputs/ppo_12288_recovery300_post10s
```

The fixed 300-seed recovery evaluation found 0 falls before crossing, 0 falls
during the additional 10 seconds, and 0 falls at the final frame. The minimum
pelvis height was 0.6980 m; maximum absolute pelvis pitch and roll were 2.064
degrees and 0.611 degrees.

The selected package and full per-episode evaluation are under
`frozen/ppo_parameters_12288_selected_20260726/`. Compact failed and
intermediate evidence is under `rl_artifacts/milestones/`; large replay
buffers, TensorBoard caches, and duplicate checkpoints are intentionally
excluded.
