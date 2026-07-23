"""Fail fast if the frozen Task 1 world-model contract has changed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT


def main() -> None:
    manifest = json.loads(
        (ROOT / "docs" / "frozen_snapshot.json").read_text(encoding="utf-8")
    )
    failures = []
    for relative, expected in manifest["files"].items():
        path = SNAPSHOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        if actual != expected:
            failures.append(f"{relative}: expected {expected}, got {actual}")
    if failures:
        raise SystemExit("Frozen Task 1 snapshot mismatch:\n" + "\n".join(failures))
    print(
        "Frozen Task 1 snapshot verified:",
        manifest["task1_tag"],
        manifest["task1_commit"],
    )


if __name__ == "__main__":
    main()
