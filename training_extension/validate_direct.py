from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from .optimize_direct import evaluate
from .basketball_env import BasketballResidualEnv


def evaluate_seed(payload):
    parameters, seed = payload
    _, records = evaluate(parameters, [seed])
    return records[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    state = json.loads(args.state.read_text(encoding="utf-8"))
    parameters = np.asarray(state["best_parameters"], dtype=np.float64)
    payloads = [
        (parameters, seed)
        for seed in range(args.seed, args.seed + args.episodes)
    ]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        records = pool.map(evaluate_seed, payloads)

    summary = {
        "episodes": len(records),
        "successes": sum(int(record["success"]) for record in records),
        "success_rate": float(np.mean([record["success"] for record in records])),
        "mean_hoop_xy_error": float(
            np.mean(
                [
                    record["crossing_xy_error"]
                    if record["crossing_xy_error"] is not None
                    else record["hoop_xy_error"]
                    for record in records
                ]
            )
        ),
        "max_hoop_xy_error": float(
            np.max(
                [
                    record["crossing_xy_error"]
                    if record["crossing_xy_error"] is not None
                    else record["hoop_xy_error"]
                    for record in records
                ]
            )
        ),
        "falls": sum(int(record["has_fallen"]) for record in records),
        "backboard_contacts": sum(
            int(record["touched_backboard"]) for record in records
        ),
        "direct_shots": sum(int(record["direct_shot"]) for record in records),
        "target": BasketballResidualEnv.target.tolist(),
        "hoop_radius": BasketballResidualEnv.hoop_radius,
        "parameters": parameters.tolist(),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output / "episodes.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
