# LEVEL03 derived basketball baseline

This directory contains a repository-relative, independently runnable
**derived baseline** built on the teammate baseline
`scripts/view_baselines_LEVEL03_v031!.py`.

The teammate file is imported and left unchanged.  The derived policy only
overrides selected keyframes to improve foot landing, reduce wrist twisting,
keep the guide hand from crossing the shooting arm, and recover with separated
arms.  It also uses a farther physical hoop centered at `(2.2, 0.0, 1.2)`.

This branch contains **no CEM and no reinforcement learning**.

## Run

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m training_extension.view_derived_baseline
```

On Windows, clone the repository to an ASCII-only path such as
`D:\projects\audi-ai-robotics-2026`. MuJoCo 3.10 may fail to open included XML
files when the absolute checkout path contains non-ASCII characters.

Evaluate 100 reset perturbations:

```powershell
python -m training_extension.evaluate_derived_baseline `
  --episodes 100 `
  --seed 17000 `
  --output outputs/derived_baseline_100
```

Recorded result for seeds `17000..17099`:

- success: `0/100`
- mean hoop-plane crossing error: `1.1032 m`
- maximum crossing error: `1.1068 m`
- backboard contacts: `0`
- falls: `0`

This intentionally imperfect result is the honest no-CEM reference. The
stacked CEM branch must report improvement against this exact contract.

Run the contract and code-quality checks:

```powershell
python -m unittest discover -s training_extension/tests -v
python -m training_extension.quality_check --smoke-episodes 2
```

## Immutable scoring contract

- target center: `(2.2, 0.0, 1.2)`
- physical 16-segment ring
- valid ball-center crossing radius: `0.10 m`
- downward plane crossing required
- no backboard contact
- release ball distance at least `1.10 m`
- release pelvis distance at least `1.20 m`
- airborne horizontal distance at least `1.00 m`
- minimum hand-to-hoop distance at least `0.45 m`
- no fall

The visual green disk has radius `0.14 m`, matching the geometric inner edge
of the ring.  The stricter `0.10 m` ball-center criterion accounts for the
`0.04 m` ball radius.

## Relationship to the CEM branch

`feature/simulation03-cem` is stacked on this branch and adds trajectory
optimization, selected parameters and optimization milestones.  Keeping CEM
separate makes the actual baseline quality visible and independently
reproducible.

## Artifacts

- `artifacts/derived_baseline_eval100_seed17000/`: machine-readable 100-seed
  evaluation
- `artifacts/quality_report_baseline.json`: compile, scene and smoke report
- `artifacts/baseline_motion_milestones/`: curated earlier motion attempts;
  these snapshots are documentation and are not imported at runtime
