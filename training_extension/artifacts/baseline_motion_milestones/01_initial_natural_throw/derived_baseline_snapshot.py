"""Training baseline derived from the teammate's unchanged LEVEL03 baseline.

Only the visibly problematic right-foot landing and upper-body throw keyframes
are adjusted.  The target, physics, scoring, walk structure, dip, and leg drive
remain unchanged.  This reference is intentionally imperfect so learning still
has a measurable throwing task.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "scripts" / "view_baselines_LEVEL03_v030!.py"


def _load_original():
    spec = importlib.util.spec_from_file_location("peer_level03_original", ORIGINAL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load original baseline: {ORIGINAL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.G1FixedBodyThrowEnv, module.OptionDBasketballPolicy


PeerEnv, OriginalPolicy = _load_original()


class TrainingBasketballPolicy(OriginalPolicy):
    """Conservative, human-readable starting point for subsequent learning."""

    def __init__(self, env):
        super().__init__(env)
        self.keyframes = copy.deepcopy(self.keyframes)

        # Reduce the exaggerated right-leg swing and keep the knee flexed as
        # the foot returns to the floor.
        self.keyframes[100].update(
            {
                "right_hip_pitch_joint": -0.72,
                "right_knee_joint": 1.05,
                "right_ankle_pitch_joint": -0.30,
            }
        )
        self.keyframes[150].update(
            {
                "right_hip_pitch_joint": -0.32,
                "right_knee_joint": 0.38,
                "right_ankle_pitch_joint": -0.16,
                "left_hip_pitch_joint": 0.28,
                "left_knee_joint": 0.36,
                "left_ankle_pitch_joint": 0.02,
            }
        )
        self.keyframes[200].update(
            {
                "right_hip_pitch_joint": 0.02,
                "right_knee_joint": 0.34,
                "right_ankle_pitch_joint": -0.15,
            }
        )

        # Rough two-hand set shot.  It is deliberately not ballistically tuned:
        # RL/trajectory optimisation must still learn launch speed and timing.
        self.keyframes[380].update(
            {
                "right_shoulder_pitch_joint": -1.05,
                "right_shoulder_roll_joint": 0.10,
                "right_elbow_joint": 1.55,
                "right_wrist_pitch_joint": -0.65,
                "left_shoulder_pitch_joint": -1.00,
                "left_shoulder_roll_joint": -0.10,
                "left_elbow_joint": 1.50,
                "left_wrist_pitch_joint": -0.45,
            }
        )
        self.keyframes[400].update(
            {
                "right_shoulder_pitch_joint": -1.35,
                "right_shoulder_roll_joint": 0.08,
                "right_elbow_joint": 0.85,
                "right_wrist_pitch_joint": -0.20,
                "left_shoulder_pitch_joint": -1.28,
                "left_shoulder_roll_joint": -0.08,
                "left_elbow_joint": 0.95,
                "left_wrist_pitch_joint": -0.15,
            }
        )
        self.keyframes[410].update(
            {
                "right_shoulder_pitch_joint": -1.60,
                "right_shoulder_roll_joint": 0.08,
                "right_elbow_joint": 0.38,
                "right_wrist_pitch_joint": 0.35,
                "left_shoulder_pitch_joint": -1.48,
                "left_shoulder_roll_joint": -0.08,
                "left_elbow_joint": 0.58,
                "left_wrist_pitch_joint": 0.12,
            }
        )
        self.keyframes[440].update(
            {
                "right_shoulder_pitch_joint": -1.62,
                "right_shoulder_roll_joint": 0.08,
                "right_elbow_joint": 0.36,
                "right_wrist_pitch_joint": 0.45,
                "left_shoulder_pitch_joint": -1.45,
                "left_shoulder_roll_joint": -0.08,
                "left_elbow_joint": 0.62,
                "left_wrist_pitch_joint": 0.15,
            }
        )
