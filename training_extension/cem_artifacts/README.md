# CEM artifacts

## Selected

`selected/` contains the chosen v17 parameter state, complete generation
history, frozen 20-seed validation, fresh 100-seed validation, quality report,
preview video and inspection frame.

The saved state is sufficient for playback; rerunning CEM is optional.

## Milestones

The milestone directories contain only curated CEM states, validation summaries
and preview videos. Runtime code never imports them.

- `01_invalid_hand_in_hoop`: rejected numerical shortcut
- `02_airborne_low_release`: rejected low sweeping release
- `03_high_release_onehand`: guide hand still too far away
- `04_twohand_gap_too_wide`: release-hand separation still over the limit
- `05_twohand_strict_success`: accepted nearer-target intermediate
- `07_far_target_openloop`: farther visible flight, footwork incomplete
- `08_second_step_fix`: improved second landing
- `09_postshot_separation`: separated recovery before final guide-hand fix

The selected v17 output is stored only once under `selected/`.
