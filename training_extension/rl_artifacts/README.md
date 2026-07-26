# RL experiment evidence

This directory preserves compact, auditable evidence from the algorithm search
without committing the large local `runs/` tree.

| Folder | Meaning |
|---|---|
| `01_td3_residual_collapse` | raw frame-level TD3 collapsed |
| `02_td3_zero_init_collapse` | zero-initialized TD3 still collapsed |
| `03_td3_filtered_collapse` | filtered TD3 still collapsed |
| `04_ppo_frame_residual_degraded` | frame-level PPO remained stable but reduced accuracy |
| `05_sac_parameter_gate` | SAC reduced one tail metric but worsened mean error |
| `06_ppo_parameter_1024` | first positive parameter-PPO checkpoint |
| `07_ppo_parameter_4096_selected` | previous selected PPO with model and media |
| `08_ppo_parameter_10752_candidate` | continued-training candidate and checkpoint sweep |

Each failed stage includes its available run metadata, monitor data, and
per-episode evaluation. The successful milestones also include the matching
model and normalization state where useful. The final selected PPO 12,288
package lives separately under `training_extension/frozen/`.

`SHA256SUMS.txt` covers every file in this directory except the manifest
itself.
