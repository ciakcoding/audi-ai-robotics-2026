from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
FROZEN = HERE / "frozen" / "ppo_parameters_12288_selected_20260726"
ARTIFACTS = HERE / "rl_artifacts"


def _verify_manifest(root: Path) -> int:
    manifest = root / "SHA256SUMS.txt"
    checked = 0
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch: {path}")
        checked += 1
    return checked


def main() -> None:
    required = [
        FROZEN / "selected_model.zip",
        FROZEN / "selected_vecnormalize.pkl",
        FROZEN / "evaluation_300_summary.json",
        FROZEN / "evaluation_300_episodes.json",
        FROZEN / "ppo_12288_seed100000.mp4",
        ARTIFACTS / "README.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing RL release files: {missing}")

    summary = json.loads(
        (FROZEN / "evaluation_300_summary.json").read_text(encoding="utf-8")
    )
    expected = {
        "episodes": 300,
        "successes": 297,
        "backboard_contacts": 0,
        "falls": 0,
    }
    for key, value in expected.items():
        if summary[key] != value:
            raise RuntimeError(f"Unexpected {key}: {summary[key]} != {value}")
    if abs(summary["mean_crossing_error"] - 0.03336485868716297) > 1e-12:
        raise RuntimeError("Selected PPO mean error does not match freeze record")

    frozen_hashes = _verify_manifest(FROZEN)
    artifact_hashes = _verify_manifest(ARTIFACTS)
    print(
        json.dumps(
            {
                "status": "PASS",
                "selected_episodes": summary["episodes"],
                "selected_successes": summary["successes"],
                "mean_crossing_error_m": summary["mean_crossing_error"],
                "max_crossing_error_m": summary["max_crossing_error"],
                "frozen_hashes_checked": frozen_hashes,
                "artifact_hashes_checked": artifact_hashes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
