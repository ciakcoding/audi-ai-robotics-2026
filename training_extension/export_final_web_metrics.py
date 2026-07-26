"""Export presentation-ready metrics for the final Level 03 PPO policy.

The primary block uses the same four metric names as the teammate's v031
scripted baseline.  The previous RL-specific metrics remain in a separate
legacy block so the web page can keep them without using them as the headline
comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate_ppo_recovery import _evaluate_seed_batch


HERE = Path(__file__).resolve().parent
DEFAULT_FROZEN = (
    HERE / "frozen" / "ppo_parameters_12288_selected_20260726"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_FROZEN / "selected_model.zip",
    )
    parser.add_argument(
        "--vecnormalize",
        type=Path,
        default=DEFAULT_FROZEN / "selected_vecnormalize.pkl",
    )
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--post-shot-seconds", type=float, default=10.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "artifacts" / "lv3_final_web_metrics.json",
    )
    args = parser.parse_args()

    record = _evaluate_seed_batch(
        str(args.model),
        str(args.vecnormalize),
        [args.seed],
        args.post_shot_seconds,
    )[0]

    pitch = record["max_torso_tilt_pitch_deg"]
    roll = record["max_torso_tilt_roll_deg"]
    yaw = record["max_torso_tilt_yaw_deg"]
    crossing_error_cm = 100.0 * record["crossing_xy_error"]

    payload = {
        "policy": "PPO parameter-residual policy, selected at 12,288 shots",
        "seed": args.seed,
        "post_shot_seconds": args.post_shot_seconds,
        "primary_display_metrics": {
            "hoop_crossing_speed": {
                "value": record["hoop_crossing_speed_m_s"],
                "unit": "m/s",
                "display": (
                    f"{record['hoop_crossing_speed_m_s']:.2f} m/s"
                ),
            },
            "max_rim_impact_force": {
                "value": record["max_rim_impact_force_n"],
                "unit": "N",
                "display": f"{record['max_rim_impact_force_n']:.2f} N",
            },
            "max_torso_tilt": {
                "pitch_deg": pitch,
                "roll_deg": roll,
                "yaw_deg": yaw,
                "display": f"{pitch:.1f}° / {roll:.1f}° / {yaw:.1f}°",
            },
            "final_ball_to_target_distance": {
                "value": record["final_ball_to_target_distance_m"],
                "unit": "m",
                "display": (
                    f"{record['final_ball_to_target_distance_m']:.2f} m"
                ),
            },
        },
        "legacy_rl_metrics": {
            "hoop_crossing_xy_error": {
                "value": crossing_error_cm,
                "unit": "cm",
                "display": f"{crossing_error_cm:.2f} cm",
            },
            "touched_backboard": record["touched_backboard"],
            "fell_over": bool(
                record["fall_before_crossing"]
                or record["fall_during_recovery"]
            ),
            "airborne_distance": {
                "value": record["airborne_horizontal_distance_m"],
                "unit": "m",
                "display": (
                    f"{record['airborne_horizontal_distance_m']:.2f} m"
                ),
            },
            "release_step": record["release_step"],
            "episode_reward": record["episode_reward"],
        },
        "display_order": [
            "hoop_crossing_speed",
            "max_rim_impact_force",
            "max_torso_tilt",
            "final_ball_to_target_distance",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
